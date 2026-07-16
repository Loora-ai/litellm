import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
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


def _get_required_speechace_environment() -> Dict[str, str]:
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


async def _send_speechace_request(
    request: Request,
    url: httpx.URL,
    headers: Dict[str, str],
) -> httpx.Response:
    raw_body = await request.body()
    async_client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": 10},
    ).client
    return await async_client.request(
        method="POST",
        url=url,
        headers=headers,
        content=raw_body,
    )


async def _log_speechace_failure(
    user_api_key_dict: UserAPIKeyAuth,
    exception: Exception,
    request_data: dict,
) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    await proxy_logging_obj.post_call_failure_hook(
        user_api_key_dict=user_api_key_dict,
        original_exception=exception,
        request_data=request_data,
    )


@router.post(
    "/v1/speechace/score",
    tags=["SpeechAce", "pass-through"],
)
async def speechace_score(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Response:
    from litellm.proxy.proxy_server import proxy_logging_obj

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

    try:
        response = await _send_speechace_request(
            request=request,
            url=url,
            headers=headers,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exception:
        await _log_speechace_failure(
            user_api_key_dict=user_api_key_dict,
            exception=exception,
            request_data={**request_data, **kwargs},
        )
        raise HTTPException(
            status_code=exception.response.status_code,
            detail=exception.response.text,
        ) from exception
    except Exception as exception:
        await _log_speechace_failure(
            user_api_key_dict=user_api_key_dict,
            exception=exception,
            request_data={**request_data, **kwargs},
        )
        raise

    content = await response.aread()
    response_body: Optional[dict] = get_response_body(response)
    passthrough_logging_payload["response_body"] = response_body
    asyncio.create_task(
        pass_through_endpoint_logging.pass_through_async_success_handler(
            httpx_response=response,
            response_body=response_body,
            url_route=str(url),
            result="",
            start_time=start_time,
            end_time=datetime.now(),
            logging_obj=logging_obj,
            cache_hit=False,
            request_body=request_data,
            custom_llm_provider="speechace",
            **kwargs,
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
