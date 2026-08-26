"""Ensure TTS / ElevenLabs writes custom_llm_provider onto litellm_params."""

from unittest.mock import MagicMock, patch


def test_speech_sets_custom_llm_provider_on_litellm_params():
    captured = {}

    mock_logging = MagicMock()

    def capture_update_environment_variables(**kwargs):
        captured["litellm_params"] = kwargs["litellm_params"]
        captured["custom_llm_provider"] = kwargs.get("custom_llm_provider")

    mock_logging.update_environment_variables.side_effect = (
        capture_update_environment_variables
    )

    mock_response = MagicMock()
    with (
        patch(
            "litellm.main.get_llm_provider",
            return_value=("eleven_turbo_v2", "elevenlabs", None, None),
        ),
        patch(
            "litellm.main.base_llm_http_handler.text_to_speech_handler",
            return_value=mock_response,
        ),
    ):
        from litellm.main import speech

        result = speech(
            model="elevenlabs/eleven_turbo_v2",
            input="hello",
            voice="Rachel",
            litellm_logging_obj=mock_logging,
        )

    assert result is mock_response
    assert captured["custom_llm_provider"] == "elevenlabs"
    assert captured["litellm_params"]["custom_llm_provider"] == "elevenlabs"
