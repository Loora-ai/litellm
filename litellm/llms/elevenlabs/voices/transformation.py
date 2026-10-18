"""
ElevenLabs "list voices" transformation.

Maps a provider-agnostic voices request to the ElevenLabs list-voices API
(GET /v2/voices). Mirrors ElevenLabsTextToSpeechConfig (same xi-api-key auth and
ELEVENLABS_API_BASE handling).

Reference: https://elevenlabs.io/docs/api-reference/voices/search
"""

from typing import TYPE_CHECKING, Any

import httpx
from httpx import Headers

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.voices.transformation import BaseVoicesConfig
from litellm.secret_managers.main import get_secret_str

from ..common_utils import ElevenLabsException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class ElevenLabsVoicesConfig(BaseVoicesConfig):
    VOICES_BASE_URL = "https://api.elevenlabs.io"
    VOICES_ENDPOINT_PATH = "/v2/voices"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        api_key = api_key or litellm.api_key or litellm.openai_key or get_secret_str("ELEVENLABS_API_KEY")

        if api_key is None:
            raise ValueError("ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable.")

        headers.update(
            {
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            }
        )

        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base_url = api_base or get_secret_str("ELEVENLABS_API_BASE") or self.VOICES_BASE_URL
        return f"{base_url.rstrip('/')}{self.VOICES_ENDPOINT_PATH}"

    def transform_voices_request(
        self,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        params = {key: value for key, value in (optional_params or {}).items() if value is not None}
        return {"params": params, "headers": {}}

    def transform_voices_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj | None",
    ) -> dict:
        return raw_response.json()

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return ElevenLabsException(message=error_message, status_code=status_code, headers=headers)
