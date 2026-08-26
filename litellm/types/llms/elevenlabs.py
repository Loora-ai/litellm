"""
Type definitions for ElevenLabs API responses and request parameters.

Reference: https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps
"""

from typing import List, Literal, TypedDict


class ElevenLabsAlignment(TypedDict):
    """
    Character-level alignment data from ElevenLabs TTS with timestamps.

    Contains arrays where each index corresponds to a character in the input text.
    """

    characters: List[str]
    character_start_times_seconds: List[float]
    character_end_times_seconds: List[float]


class ElevenLabsNormalizedAlignment(TypedDict):
    """
    Normalized alignment data with consistent timing across the audio.
    """

    characters: List[str]
    character_start_times_seconds: List[float]
    character_end_times_seconds: List[float]


class ElevenLabsTextToSpeechWithTimestampsResponse(TypedDict, total=False):
    """
    Response from ElevenLabs TTS with-timestamps endpoint.

    This response is returned when with_timestamps=True is passed to the speech() call.
    Contains base64-encoded audio and character-level timing information.

    Usage:
        response = litellm.speech(
            model="elevenlabs/eleven_turbo_v2",
            input="Hello world",
            voice="alloy",
            extra_body={"with_timestamps": True}
        )
        # response is a dict with audio_base64 and alignment data
        audio_bytes = base64.b64decode(response["audio_base64"])
        alignment = response["alignment"]
    """

    audio_base64: str
    alignment: ElevenLabsAlignment
    normalized_alignment: ElevenLabsNormalizedAlignment


class ElevenLabsPronunciationDictionaryLocator(TypedDict):
    """
    Locator for a specific version of a pronunciation dictionary.

    Reference: https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries
    """

    pronunciation_dictionary_id: str
    version_id: str


class ElevenLabsVoiceSettings(TypedDict, total=False):
    """
    Voice settings for ElevenLabs TTS.

    All values are optional and have sensible defaults.
    """

    stability: float  # 0.0-1.0: Higher = more consistent, lower = more expressive
    similarity_boost: float  # 0.0-1.0: Higher = closer to original voice
    style: float  # 0.0-1.0: Style exaggeration (v2 models only)
    use_speaker_boost: bool  # Boost voice clarity and similarity
    speed: float  # 0.25-4.0: Speech speed multiplier


# Text normalization options for ElevenLabs
ElevenLabsTextNormalization = Literal["auto", "on", "off"]


class ElevenLabsExtraBody(TypedDict, total=False):
    """
    Extra body parameters for ElevenLabs TTS requests via litellm.

    Usage:
        response = litellm.speech(
            model="elevenlabs/eleven_turbo_v2",
            input="Hello world",
            voice="alloy",
            extra_body={
                "with_timestamps": True,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "speed": 1.0,
                },
                "pronunciation_dictionary_locators": [
                    {
                        "pronunciation_dictionary_id": "dict_id",
                        "version_id": "version_id",
                    }
                ],
                "apply_text_normalization": "auto",
            }
        )
    """

    with_timestamps: bool
    voice_settings: ElevenLabsVoiceSettings
    pronunciation_dictionary_locators: List[ElevenLabsPronunciationDictionaryLocator]
    apply_text_normalization: ElevenLabsTextNormalization
