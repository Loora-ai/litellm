import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import Headers

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.speechace_connector import (
    _build_speechace_url,
    _get_required_speechace_environment,
    _send_speechace_request,
    router,
    speechace_score,
    speechace_user_api_key_auth,
)


def _set_speechace_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SPEECHACE_TARGET_URL",
        "https://api4.speechace.com/api/scoring/text/v9/json",
    )
    monkeypatch.setenv("SPEECHACE_API_KEY", "speechace-key")
    monkeypatch.setenv("SPEECHACE_USER_ID", "default-user")


def test_speechace_route_is_authenticated_and_allowed_for_llm_keys() -> None:
    speechace_route = next(
        route for route in router.routes if route.path == "/v1/speechace/score"
    )
    dependency_calls = [
        dependency.call for dependency in speechace_route.dependant.dependencies
    ]

    assert "/v1/speechace/score" in LiteLLMRoutes.llm_api_routes.value
    assert speechace_user_api_key_auth in dependency_calls


def test_speechace_request_query_parameters_override_defaults() -> None:
    url = _build_speechace_url(
        target_url="https://speechace.example/score",
        request_query_params={"user_id": "request-user", "dialect": "en-us"},
        default_query_params={
            "key": "speechace-key",
            "user_id": "default-user",
        },
    )

    assert dict(url.params) == {
        "key": "speechace-key",
        "user_id": "request-user",
        "dialect": "en-us",
    }


@pytest.mark.asyncio
async def test_speechace_score_streams_multipart_without_auth_reading_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_speechace_environment(monkeypatch)
    upstream_response = httpx.Response(
        status_code=200,
        content=b'{"status":"success"}',
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://speechace.example/score"),
    )
    auth_read_body = False

    async def return_pre_call_data(**kwargs):
        return kwargs["data"]

    async def authenticate_without_reading_body(request: Request) -> UserAPIKeyAuth:
        nonlocal auth_read_body
        if hasattr(request, "_body") and request._body is not None:
            auth_read_body = True
        return UserAPIKeyAuth()

    async def capture_send_request(*, request: Request, url, headers):
        body = b""
        async for chunk in request.stream():
            body += chunk
        assert b"raw audio bytes" in body
        assert "authorization" not in headers
        assert "x-litellm-metadata" not in headers
        assert headers["content-type"].startswith("multipart/form-data; boundary=")
        assert dict(url.params) == {
            "key": "speechace-key",
            "user_id": "request-user",
            "dialect": "en-us",
        }
        return upstream_response

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[speechace_user_api_key_auth] = (
        authenticate_without_reading_body
    )

    with (
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
            new=AsyncMock(side_effect=return_pre_call_data),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector._send_speechace_request",
            new=AsyncMock(side_effect=capture_send_request),
        ) as mock_send_request,
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector.pass_through_endpoint_logging.pass_through_async_success_handler",
            new=AsyncMock(),
        ) as mock_success_handler,
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://proxy.example",
        ) as client:
            response = await client.post(
                "/v1/speechace/score",
                params={"user_id": "request-user", "dialect": "en-us"},
                headers={
                    "authorization": "Bearer litellm-key",
                    "x-litellm-metadata": '{"spend_logs_metadata":{"service":"application-backend"}}',
                },
                data={"text": "hello"},
                files={
                    "user_audio_file": (
                        "speechace.wav",
                        b"raw audio bytes",
                        "audio/wav",
                    )
                },
            )
        await asyncio.sleep(0)

    assert response.content == b'{"status":"success"}'
    assert auth_read_body is False
    assert mock_send_request.await_count == 1
    assert mock_success_handler.await_args is not None
    success_kwargs = mock_success_handler.await_args.kwargs
    assert success_kwargs["custom_llm_provider"] == "speechace"
    assert success_kwargs["litellm_params"]["metadata"]["spend_logs_metadata"] == {
        "service": "application-backend"
    }


@pytest.mark.asyncio
async def test_send_speechace_request_streams_body_chunks() -> None:
    request = MagicMock(spec=Request)

    async def stream_chunks():
        yield b"chunk-one"
        yield b"chunk-two"

    request.stream = MagicMock(return_value=stream_chunks())
    expected_response = MagicMock(spec=httpx.Response)
    async_client = MagicMock(spec=httpx.AsyncClient)
    async_client.request = AsyncMock(return_value=expected_response)

    client_wrapper = MagicMock()
    client_wrapper.client = async_client
    with patch(
        "litellm.proxy.pass_through_endpoints.speechace_connector.get_async_httpx_client",
        return_value=client_wrapper,
    ):
        response = await _send_speechace_request(
            request=request,
            url=httpx.URL("https://speechace.example/score?user_id=request-user"),
            headers={
                "content-type": "multipart/form-data; boundary=speechace-boundary"
            },
        )

    assert response is expected_response
    call_kwargs = async_client.request.await_args.kwargs
    streamed_body = b""
    async for chunk in call_kwargs["content"]:
        streamed_body += chunk
    assert streamed_body == b"chunk-onechunk-two"
    assert "json" not in call_kwargs
    assert "files" not in call_kwargs


@pytest.mark.asyncio
async def test_speechace_score_rejects_invalid_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_speechace_environment(monkeypatch)
    request = MagicMock(spec=Request)
    request.headers = Headers({"content-type": "application/json"})
    request.stream = MagicMock()

    with pytest.raises(HTTPException) as exception:
        await speechace_score(
            request=request,
            user_api_key_dict=UserAPIKeyAuth(),
        )

    assert exception.value.status_code == 415
    request.stream.assert_not_called()


def test_speechace_environment_requires_all_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPEECHACE_TARGET_URL", raising=False)
    monkeypatch.delenv("SPEECHACE_API_KEY", raising=False)
    monkeypatch.delenv("SPEECHACE_USER_ID", raising=False)

    with pytest.raises(HTTPException) as exception:
        _get_required_speechace_environment()

    assert exception.value.status_code == 503
    assert "SPEECHACE_TARGET_URL" in exception.value.detail
    assert "SPEECHACE_API_KEY" in exception.value.detail
    assert "SPEECHACE_USER_ID" in exception.value.detail


@pytest.mark.asyncio
async def test_speechace_score_propagates_upstream_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_speechace_environment(monkeypatch)
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = httpx.URL("https://proxy.example/v1/speechace/score")
    request.query_params = {"user_id": "request-user"}
    request.headers = Headers(
        {"content-type": "multipart/form-data; boundary=speechace-boundary"}
    )
    upstream_response = httpx.Response(
        status_code=502,
        content=b"SpeechAce unavailable",
        request=httpx.Request("POST", "https://speechace.example/score"),
    )

    async def return_pre_call_data(**kwargs):
        return kwargs["data"]

    with (
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
            new=AsyncMock(side_effect=return_pre_call_data),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector._send_speechace_request",
            new=AsyncMock(return_value=upstream_response),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector._log_speechace_failure",
            new=AsyncMock(),
        ) as mock_failure_handler,
    ):
        with pytest.raises(HTTPException) as exception:
            await speechace_score(
                request=request,
                user_api_key_dict=UserAPIKeyAuth(),
            )

    assert exception.value.status_code == 502
    mock_failure_handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_speechace_user_api_key_auth_skips_body_read() -> None:
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.headers = Headers({"authorization": "Bearer sk-test"})
    request.url = httpx.URL("https://proxy.example/v1/speechace/score")
    request.body = AsyncMock(side_effect=AssertionError("body must not be read"))
    request.form = AsyncMock(side_effect=AssertionError("form must not be read"))

    auth_obj = UserAPIKeyAuth(api_key="sk-test")

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector._ensure_parent_otel_span_on_request_state"
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector.get_request_route",
            return_value="/v1/speechace/score",
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector._user_api_key_auth_builder",
            new=AsyncMock(return_value=auth_obj),
        ) as mock_builder,
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector.RouteChecks.should_call_route"
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector._run_centralized_common_checks",
            new=AsyncMock(),
        ) as mock_common_checks,
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector.normalize_request_route",
            return_value="/v1/speechace/score",
        ),
    ):
        result = await speechace_user_api_key_auth(
            request=request,
            api_key="Bearer sk-test",
            azure_api_key_header_value="",
            anthropic_api_key_header_value=None,
            google_ai_studio_api_key_header_value=None,
            azure_apim_header_value=None,
            custom_litellm_key_header_value=None,
        )

    assert result is auth_obj
    assert mock_builder.await_args is not None
    assert mock_builder.await_args.kwargs["request_data"] == {}
    assert mock_common_checks.await_args is not None
    assert mock_common_checks.await_args.kwargs["request_data"] == {}
    request.body.assert_not_awaited()
    request.form.assert_not_awaited()
