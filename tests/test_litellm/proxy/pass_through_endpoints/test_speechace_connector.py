import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request
from starlette.datastructures import Headers

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints.speechace_connector import (
    _build_speechace_url,
    _get_required_speechace_environment,
    _send_speechace_request,
    router,
    speechace_score,
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

    assert "/v1/speechace/score" in LiteLLMRoutes.llm_api_routes.value
    assert any(
        dependency.call is user_api_key_auth
        for dependency in speechace_route.dependant.dependencies
    )


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
async def test_speechace_score_forwards_raw_multipart_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_speechace_environment(monkeypatch)
    raw_body = b"--speechace-boundary\r\nraw audio bytes\r\n--speechace-boundary--"
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = httpx.URL("https://proxy.example/v1/speechace/score")
    request.query_params = {"user_id": "request-user", "dialect": "en-us"}
    request.headers = Headers(
        {
            "content-type": "multipart/form-data; boundary=speechace-boundary",
            "accept": "application/json",
            "authorization": "Bearer litellm-key",
            "x-litellm-metadata": '{"spend_logs_metadata":{"service":"application-backend"}}',
        }
    )
    request.body = AsyncMock(return_value=raw_body)
    upstream_response = httpx.Response(
        status_code=200,
        content=b'{"status":"success"}',
        headers={"content-type": "application/json"},
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
        ) as mock_send_request,
        patch(
            "litellm.proxy.pass_through_endpoints.speechace_connector.pass_through_endpoint_logging.pass_through_async_success_handler",
            new=AsyncMock(),
        ) as mock_success_handler,
    ):
        response = await speechace_score(
            request=request,
            user_api_key_dict=UserAPIKeyAuth(),
        )
        await asyncio.sleep(0)

    assert response.body == b'{"status":"success"}'
    assert mock_send_request.await_args is not None
    send_kwargs = mock_send_request.await_args.kwargs
    assert send_kwargs["headers"] == {
        "content-type": "multipart/form-data; boundary=speechace-boundary",
        "accept": "application/json",
    }
    assert "authorization" not in send_kwargs["headers"]
    assert "x-litellm-metadata" not in send_kwargs["headers"]
    assert dict(send_kwargs["url"].params) == {
        "key": "speechace-key",
        "user_id": "request-user",
        "dialect": "en-us",
    }
    assert mock_success_handler.await_args is not None
    success_kwargs = mock_success_handler.await_args.kwargs
    assert success_kwargs["custom_llm_provider"] == "speechace"
    assert success_kwargs["litellm_params"]["metadata"]["spend_logs_metadata"] == {
        "service": "application-backend"
    }


@pytest.mark.asyncio
async def test_raw_body_request_skips_multipart_parsing() -> None:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = Headers(
        {"content-type": "multipart/form-data; boundary=speechace-boundary"}
    )
    raw_body = b"unchanged multipart body"
    request.body = AsyncMock(return_value=raw_body)
    request.form = AsyncMock()
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
            headers={"content-type": request.headers["content-type"]},
        )

    assert response is expected_response
    request.body.assert_awaited_once()
    request.form.assert_not_awaited()
    call_kwargs = async_client.request.await_args.kwargs
    assert call_kwargs["content"] == raw_body
    assert "json" not in call_kwargs
    assert "files" not in call_kwargs


@pytest.mark.asyncio
async def test_speechace_score_rejects_invalid_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_speechace_environment(monkeypatch)
    request = MagicMock(spec=Request)
    request.headers = Headers({"content-type": "application/json"})
    request.body = AsyncMock()

    with pytest.raises(HTTPException) as exception:
        await speechace_score(
            request=request,
            user_api_key_dict=UserAPIKeyAuth(),
        )

    assert exception.value.status_code == 415
    request.body.assert_not_awaited()


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
    request.body = AsyncMock(return_value=b"multipart body")
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
