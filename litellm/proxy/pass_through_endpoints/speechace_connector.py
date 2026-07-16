import asyncio
import os
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Union
from urllib.parse import urlencode

import fastapi
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id
from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler
from litellm.proxy.auth.auth_utils import (
    get_end_user_id_from_request_body,
    get_request_route,
    normalize_request_route,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import (
    _ensure_parent_otel_span_on_request_state,
    _run_centralized_common_checks,
    _user_api_key_auth_builder,
    anthropic_api_key_header,
    api_key_header,
    azure_api_key_header,
    azure_apim_header,
    custom_litellm_key_header,
    google_ai_studio_api_key_header,
)
from litellm.proxy.common_utils.http_parsing_utils import (
    _safe_get_request_headers,
    populate_request_with_path_params,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
    get_response_body,
    pass_through_endpoint_logging,
)
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)

router = APIRouter()
_cached_speechace_environment: Optional[Dict[str, str]] = None


def _get_required_speechace_environment() -> Dict[str, str]:
    global _cached_speechace_environment
    if _cached_speechace_environment is not None:
        return _cached_speechace_environment

    environment = {
        "target_url": os.getenv("SPEECHACE_TARGET_URL", ""),
        "api_key": os.getenv("SPEECHACE_API_KEY", ""),
        "user_id": os.getenv("SPEECHACE_USER_ID", ""),
    }
    missing_variables = [
        variable_name
        for variable_name, value in {
            "SPEECHACE_TARGET_URL": environment["target_url"],
            "SPEECHACE_API_KEY": environment["api_key"],
            "SPEECHACE_USER_ID": environment["user_id"],
        }.items()
        if not value
    ]
    if missing_variables:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Missing SpeechAce configuration: {', '.join(missing_variables)}",
        )
    _cached_speechace_environment = environment
    return environment


def _build_speechace_url(
    target_url: str,
    request_query_params: Dict[str, str],
    default_query_params: Dict[str, Union[str, List[str]]],
) -> httpx.URL:
    url = httpx.URL(target_url)
    merged_query_params = HttpPassThroughEndpointHelpers.get_merged_query_parameters(
        existing_url=url,
        request_query_params=request_query_params,
        default_query_params=default_query_params,
    )
    return url.copy_with(query=urlencode(merged_query_params).encode("ascii"))


async def speechace_user_api_key_auth(
    request: Request,
    api_key: str = fastapi.Security(api_key_header),
    azure_api_key_header_value: str = fastapi.Security(azure_api_key_header),
    anthropic_api_key_header_value: Optional[str] = fastapi.Security(
        anthropic_api_key_header
    ),
    google_ai_studio_api_key_header_value: Optional[str] = fastapi.Security(
        google_ai_studio_api_key_header
    ),
    azure_apim_header_value: Optional[str] = fastapi.Security(azure_apim_header),
    custom_litellm_key_header_value: Optional[str] = fastapi.Security(
        custom_litellm_key_header
    ),
) -> UserAPIKeyAuth:
    """
    LiteLLM key auth for SpeechAce that never consumes the request body.

    Multipart audio must remain unread so it can stream upstream.
    Key/team/budget/route checks still run against headers + empty body data.
    End-user attribution uses ``x-litellm-end-user-id`` when present.
    """
    _ensure_parent_otel_span_on_request_state(request)

    request_data = populate_request_with_path_params(
        request_data={},
        request=request,
    )
    route: str = get_request_route(request=request)

    user_api_key_auth_obj = await _user_api_key_auth_builder(
        request=request,
        api_key=api_key,
        azure_api_key_header=azure_api_key_header_value,
        anthropic_api_key_header=anthropic_api_key_header_value,
        google_ai_studio_api_key_header=google_ai_studio_api_key_header_value,
        azure_apim_header=azure_apim_header_value,
        request_data=request_data,
        custom_litellm_key_header=custom_litellm_key_header_value,
    )
    user_api_key_auth_obj.budget_reservation = None

    RouteChecks.should_call_route(route=route, valid_token=user_api_key_auth_obj)

    try:
        await _run_centralized_common_checks(
            user_api_key_auth_obj=user_api_key_auth_obj,
            request=request,
            request_data=request_data,
            route=route,
        )
    except Exception as e:
        return await UserAPIKeyAuthExceptionHandler._handle_authentication_error(
            e=e,
            request=request,
            request_data=request_data,
            route=route,
            parent_otel_span=user_api_key_auth_obj.parent_otel_span,
            api_key=api_key,
        )

    if user_api_key_auth_obj.end_user_id is None:
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        raw_end_user_id = get_end_user_id_from_request_body(
            request_data, _safe_get_request_headers(request)
        )
        if raw_end_user_id is not None:
            resolved_end_user_id = await resolve_and_validate_end_user_id(
                raw_end_user_id=raw_end_user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth_obj.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
                route=route,
            )
            if resolved_end_user_id is not None:
                user_api_key_auth_obj.end_user_id = resolved_end_user_id

    user_api_key_auth_obj.request_route = normalize_request_route(route)
    return user_api_key_auth_obj


async def _stream_request_body(request: Request) -> AsyncIterator[bytes]:
    async for chunk in request.stream():
        yield chunk


async def _send_speechace_request(
    request: Request,
    url: httpx.URL,
    headers: Dict[str, str],
) -> httpx.Response:
    async_client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": 10},
    ).client
    return await async_client.request(
        method="POST",
        url=url,
        headers=headers,
        content=_stream_request_body(request),
    )


async def _build_speechace_logging_context(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    url: httpx.URL,
    headers: Dict[str, str],
    litellm_call_id: str,
    start_time: datetime,
) -> tuple[Logging, dict, dict, PassthroughStandardLoggingPayload]:
    from litellm.proxy.proxy_server import proxy_logging_obj

    logging_obj = Logging(
        model="speechace/score",
        messages=[{"role": "user", "content": ""}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=start_time,
        litellm_call_id=litellm_call_id,
        function_id="speechace-score",
    )
    request_data = {"litellm_logging_obj": logging_obj}
    request_data = await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        data=request_data,
        call_type="pass_through_endpoint",
    )
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url=str(url),
        request_body=request_data,
        request_method="POST",
    )
    kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=user_api_key_dict,
        passthrough_logging_payload=passthrough_logging_payload,
        logging_obj=logging_obj,
        _parsed_body=request_data,
        litellm_call_id=litellm_call_id,
    )
    # deployment_state / deployment_* metrics read api_provider from
    # litellm_params.custom_llm_provider; request latency uses the success
    # handler's custom_llm_provider field instead.
    kwargs["litellm_params"]["custom_llm_provider"] = "speechace"
    logging_obj.model_call_details["custom_llm_provider"] = "speechace"
    logging_obj.model_call_details["litellm_params"] = kwargs["litellm_params"]
    logging_obj.update_environment_variables(
        model="speechace/score",
        user="unknown",
        optional_params={},
        litellm_params=kwargs["litellm_params"],
        call_type="pass_through_endpoint",
    )
    logging_obj.pre_call(
        input=[{"role": "user", "content": ""}],
        api_key="",
        additional_args={
            "complete_input_dict": request_data,
            "api_base": str(url),
            "headers": headers,
        },
    )
    return logging_obj, request_data, kwargs, passthrough_logging_payload


async def _log_speechace_success(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    response: httpx.Response,
    response_body: Optional[dict],
    url: httpx.URL,
    headers: Dict[str, str],
    litellm_call_id: str,
    start_time: datetime,
    end_time: datetime,
) -> None:
    (
        logging_obj,
        request_data,
        kwargs,
        passthrough_logging_payload,
    ) = await _build_speechace_logging_context(
        request=request,
        user_api_key_dict=user_api_key_dict,
        url=url,
        headers=headers,
        litellm_call_id=litellm_call_id,
        start_time=start_time,
    )
    passthrough_logging_payload["response_body"] = response_body
    await pass_through_endpoint_logging.pass_through_async_success_handler(
        httpx_response=response,
        response_body=response_body,
        url_route=str(url),
        result="",
        start_time=start_time,
        end_time=end_time,
        logging_obj=logging_obj,
        cache_hit=False,
        request_body=request_data,
        custom_llm_provider="speechace",
        **kwargs,
    )


async def _log_speechace_failure(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    exception: Exception,
    url: httpx.URL,
    headers: Dict[str, str],
    litellm_call_id: str,
    start_time: datetime,
) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    (
        logging_obj,
        request_data,
        kwargs,
        _,
    ) = await _build_speechace_logging_context(
        request=request,
        user_api_key_dict=user_api_key_dict,
        url=url,
        headers=headers,
        litellm_call_id=litellm_call_id,
        start_time=start_time,
    )
    await proxy_logging_obj.post_call_failure_hook(
        user_api_key_dict=user_api_key_dict,
        original_exception=exception,
        request_data={**request_data, **kwargs},
    )


@router.post(
    "/v1/speechace/score",
    tags=["SpeechAce", "pass-through"],
)
async def speechace_score(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(speechace_user_api_key_auth),
) -> Response:
    # Latency-first hot path: authenticate (dependency) → upstream → respond.
    # LiteLLM logging/metrics are deferred off the critical path.
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="SpeechAce scoring requires multipart/form-data with a boundary",
        )

    environment = _get_required_speechace_environment()
    headers = {"content-type": content_type}
    accept = request.headers.get("accept")
    if accept:
        headers["accept"] = accept
    content_length = request.headers.get("content-length")
    if content_length:
        headers["content-length"] = content_length

    url = _build_speechace_url(
        target_url=environment["target_url"],
        request_query_params=dict(request.query_params),
        default_query_params={
            "key": environment["api_key"],
            "user_id": environment["user_id"],
        },
    )

    litellm_call_id = str(uuid.uuid4())
    start_time = datetime.now()

    try:
        response = await _send_speechace_request(
            request=request,
            url=url,
            headers=headers,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exception:
        await _log_speechace_failure(
            request=request,
            user_api_key_dict=user_api_key_dict,
            exception=exception,
            url=url,
            headers=headers,
            litellm_call_id=litellm_call_id,
            start_time=start_time,
        )
        raise HTTPException(
            status_code=exception.response.status_code,
            detail=exception.response.text,
        ) from exception
    except Exception as exception:
        await _log_speechace_failure(
            request=request,
            user_api_key_dict=user_api_key_dict,
            exception=exception,
            url=url,
            headers=headers,
            litellm_call_id=litellm_call_id,
            start_time=start_time,
        )
        raise

    content = await response.aread()
    end_time = datetime.now()
    response_body: Optional[dict] = get_response_body(response)
    asyncio.create_task(
        _log_speechace_success(
            request=request,
            user_api_key_dict=user_api_key_dict,
            response=response,
            response_body=response_body,
            url=url,
            headers=headers,
            litellm_call_id=litellm_call_id,
            start_time=start_time,
            end_time=end_time,
        )
    )

    return Response(
        content=content,
        status_code=response.status_code,
        headers=HttpPassThroughEndpointHelpers.get_response_headers(
            headers=response.headers,
            litellm_call_id=litellm_call_id,
        ),
    )
