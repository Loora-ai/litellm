from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.elevenlabs.text_to_speech.transformation import (
    ElevenLabsTextToSpeechConfig,
)


def test_should_encode_elevenlabs_voice_id_path_segment():
    config = ElevenLabsTextToSpeechConfig()

    url = config.get_complete_url(
        model="elevenlabs/tts",
        api_base="https://api.elevenlabs.io",
        litellm_params={
            config.ELEVENLABS_VOICE_ID_KEY: "voice/../../models?x=1#frag",
        },
    )

    assert (
        url
        == "https://api.elevenlabs.io/v1/text-to-speech/voice%2F..%2F..%2Fmodels%3Fx%3D1%23frag"
    )


def test_should_reject_dot_segment_elevenlabs_voice_id():
    config = ElevenLabsTextToSpeechConfig()

    with pytest.raises(ValueError, match="voice_id cannot be a dot path segment"):
        config.get_complete_url(
            model="elevenlabs/tts",
            api_base="https://api.elevenlabs.io",
            litellm_params={config.ELEVENLABS_VOICE_ID_KEY: ".."},
        )


def test_with_timestamps_parameter_appends_endpoint_suffix():
    """Test that with_timestamps=True appends /with-timestamps to the URL."""
    config = ElevenLabsTextToSpeechConfig()

    url = config.get_complete_url(
        model="elevenlabs/eleven_turbo_v2",
        api_base="https://api.elevenlabs.io",
        litellm_params={
            config.ELEVENLABS_VOICE_ID_KEY: "21m00Tcm4TlvDq8ikWAM",
            config.ELEVENLABS_WITH_TIMESTAMPS_KEY: True,
        },
    )

    assert "/with-timestamps" in url
    assert url == "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM/with-timestamps"


def test_without_timestamps_parameter_no_suffix():
    """Test that without with_timestamps, the URL has no /with-timestamps suffix."""
    config = ElevenLabsTextToSpeechConfig()

    url = config.get_complete_url(
        model="elevenlabs/eleven_turbo_v2",
        api_base="https://api.elevenlabs.io",
        litellm_params={
            config.ELEVENLABS_VOICE_ID_KEY: "21m00Tcm4TlvDq8ikWAM",
            config.ELEVENLABS_WITH_TIMESTAMPS_KEY: False,
        },
    )

    assert "/with-timestamps" not in url
    assert url == "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"


def test_transform_response_json_with_timestamps():
    """Test that JSON responses (with-timestamps) are returned as dict."""
    config = ElevenLabsTextToSpeechConfig()

    json_response_data = {
        "audio_base64": "SGVsbG8gV29ybGQ=",
        "alignment": {
            "characters": ["H", "e", "l", "l", "o"],
            "character_start_times_seconds": [0.0, 0.1, 0.2, 0.3, 0.4],
            "character_end_times_seconds": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        "normalized_alignment": {
            "characters": ["H", "e", "l", "l", "o"],
            "character_start_times_seconds": [0.0, 0.1, 0.2, 0.3, 0.4],
            "character_end_times_seconds": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = json_response_data

    mock_logging_obj = MagicMock()

    result = config.transform_text_to_speech_response(
        model="elevenlabs/eleven_turbo_v2",
        raw_response=mock_response,
        logging_obj=mock_logging_obj,
    )

    assert isinstance(result, dict)
    assert result["audio_base64"] == "SGVsbG8gV29ybGQ="
    assert "alignment" in result
    assert "normalized_alignment" in result


def test_transform_response_binary_audio():
    """Test that binary responses are wrapped in HttpxBinaryResponseContent."""
    config = ElevenLabsTextToSpeechConfig()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.headers = {"content-type": "audio/mpeg"}

    mock_logging_obj = MagicMock()

    result = config.transform_text_to_speech_response(
        model="elevenlabs/eleven_turbo_v2",
        raw_response=mock_response,
        logging_obj=mock_logging_obj,
    )

    assert not isinstance(result, dict)


def test_map_openai_params_extracts_with_timestamps():
    """Test that with_timestamps is extracted from optional_params and passed through kwargs."""
    config = ElevenLabsTextToSpeechConfig()

    kwargs: dict = {}
    optional_params = {"with_timestamps": True}

    voice, mapped_params = config.map_openai_params(
        model="elevenlabs/eleven_turbo_v2",
        optional_params=optional_params,
        voice="alloy",
        drop_params=False,
        kwargs=kwargs,
    )

    assert kwargs.get(config.ELEVENLABS_WITH_TIMESTAMPS_KEY) is True
    assert "with_timestamps" not in mapped_params


def test_transform_request_with_pronunciation_dictionary():
    """Test that pronunciation_dictionary_locators is passed through to request body."""
    config = ElevenLabsTextToSpeechConfig()

    optional_params = {
        "extra_body": {
            "pronunciation_dictionary_locators": [
                {
                    "pronunciation_dictionary_id": "dict_123",
                    "version_id": "version_456",
                }
            ],
            "apply_text_normalization": "auto",
        }
    }

    request_data = config.transform_text_to_speech_request(
        model="eleven_turbo_v2",
        input="Hello world",
        voice="21m00Tcm4TlvDq8ikWAM",
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    body = request_data["dict_body"]
    assert "pronunciation_dictionary_locators" in body
    assert len(body["pronunciation_dictionary_locators"]) == 1
    assert body["pronunciation_dictionary_locators"][0]["pronunciation_dictionary_id"] == "dict_123"
    assert body["pronunciation_dictionary_locators"][0]["version_id"] == "version_456"
    assert body["apply_text_normalization"] == "auto"


def test_transform_request_with_voice_settings():
    """Test that voice_settings is passed through to request body."""
    config = ElevenLabsTextToSpeechConfig()

    optional_params = {
        "extra_body": {
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
                "speed": 1.2,
            }
        }
    }

    request_data = config.transform_text_to_speech_request(
        model="eleven_turbo_v2",
        input="Hello world",
        voice="21m00Tcm4TlvDq8ikWAM",
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    body = request_data["dict_body"]
    assert "voice_settings" in body
    assert body["voice_settings"]["stability"] == 0.5
    assert body["voice_settings"]["similarity_boost"] == 0.75
    assert body["voice_settings"]["speed"] == 1.2


def test_voice_settings_direct_param():
    """Test that voice_settings can be passed directly in kwargs (not in extra_body)."""
    config = ElevenLabsTextToSpeechConfig()

    kwargs: dict = {
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
    }
    optional_params: dict = {}

    voice, mapped_params = config.map_openai_params(
        model="elevenlabs/eleven_turbo_v2",
        optional_params=optional_params,
        voice="alloy",
        drop_params=False,
        kwargs=kwargs,
    )

    assert "voice_settings" in mapped_params
    assert mapped_params["voice_settings"]["stability"] == 0.5
    assert mapped_params["voice_settings"]["similarity_boost"] == 0.75


def test_speed_merges_with_direct_voice_settings():
    """Test that speed param is merged into directly passed voice_settings."""
    config = ElevenLabsTextToSpeechConfig()

    kwargs: dict = {
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
    }
    optional_params: dict = {"speed": 1.2}

    voice, mapped_params = config.map_openai_params(
        model="elevenlabs/eleven_turbo_v2",
        optional_params=optional_params,
        voice="alloy",
        drop_params=False,
        kwargs=kwargs,
    )

    assert "voice_settings" in mapped_params
    assert mapped_params["voice_settings"]["stability"] == 0.5
    assert mapped_params["voice_settings"]["similarity_boost"] == 0.75
    assert mapped_params["voice_settings"]["speed"] == 1.2


def test_voice_settings_merge_from_multiple_sources():
    """Test that voice_settings from kwargs and extra_body are merged."""
    config = ElevenLabsTextToSpeechConfig()

    kwargs: dict = {
        "voice_settings": {
            "stability": 0.5,
        },
        "extra_body": {
            "voice_settings": {
                "similarity_boost": 0.75,
            }
        }
    }
    optional_params: dict = {"speed": 1.2}

    voice, mapped_params = config.map_openai_params(
        model="elevenlabs/eleven_turbo_v2",
        optional_params=optional_params,
        voice="alloy",
        drop_params=False,
        kwargs=kwargs,
    )

    assert "voice_settings" in mapped_params
    assert mapped_params["voice_settings"]["stability"] == 0.5
    assert mapped_params["voice_settings"]["similarity_boost"] == 0.75
    assert mapped_params["voice_settings"]["speed"] == 1.2


def test_map_openai_params_extracts_with_timestamps_from_extra_body():
    """Test that with_timestamps is extracted from extra_body in kwargs (proxy flow)."""
    config = ElevenLabsTextToSpeechConfig()

    kwargs: dict = {
        "extra_body": {
            "with_timestamps": True,
            "voice_settings": {"stability": 0.5},
        }
    }
    optional_params: dict = {}

    voice, mapped_params = config.map_openai_params(
        model="elevenlabs/eleven_turbo_v2",
        optional_params=optional_params,
        voice="alloy",
        drop_params=False,
        kwargs=kwargs,
    )

    assert kwargs.get(config.ELEVENLABS_WITH_TIMESTAMPS_KEY) is True
