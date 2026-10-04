#!/usr/bin/env python3
"""
Test script for ElevenLabs TTS with timestamps support.

Usage:
    # Set your API key
    export ELEVENLABS_API_KEY=your_api_key

    # Run directly with litellm
    python scripts/test_elevenlabs_tts.py

    # Or via proxy (start proxy first: litellm --config proxy_server_config.yaml)
    python scripts/test_elevenlabs_tts.py --proxy http://localhost:4000
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for local litellm import
sys.path.insert(0, str(Path(__file__).parent.parent))

import litellm


def test_regular_tts(model: str, voice: str, api_key: str | None = None):
    """Test regular TTS - returns binary audio."""
    print("\n" + "=" * 60)
    print("TEST 1: Regular TTS (binary audio)")
    print("=" * 60)

    response = litellm.speech(
        model=model,
        input="Hello! This is a test of ElevenLabs text to speech.",
        voice=voice,
        api_key=api_key,
    )

    # Save the audio
    output_path = Path("test_output_regular.mp3")
    with open(output_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

    print(f"✓ Audio saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")


def test_tts_with_timestamps(model: str, voice: str, api_key: str | None = None):
    """Test TTS with timestamps - returns JSON with alignment data."""
    print("\n" + "=" * 60)
    print("TEST 2: TTS with Timestamps (JSON response)")
    print("=" * 60)

    response = litellm.speech(
        model=model,
        input="Hello! This is a test.",
        voice=voice,
        api_key=api_key,
        extra_body={"with_timestamps": True},
    )

    assert isinstance(response, dict), f"Expected dict, got {type(response)}"

    print("✓ Response is a dict (as expected)")
    print(f"  Keys: {list(response.keys())}")

    # Decode and save audio
    if "audio_base64" in response:
        audio_bytes = base64.b64decode(response["audio_base64"])
        output_path = Path("test_output_timestamps.mp3")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"✓ Audio saved to: {output_path}")
        print(f"  File size: {len(audio_bytes):,} bytes")

    # Show alignment data
    if "alignment" in response:
        alignment = response["alignment"]
        print("\n  Alignment data (first 10 characters):")
        chars = alignment.get("characters", [])[:10]
        starts = alignment.get("character_start_times_seconds", [])[:10]
        ends = alignment.get("character_end_times_seconds", [])[:10]

        for i, (char, start, end) in enumerate(zip(chars, starts, ends)):
            print(f"    [{i}] '{char}': {start:.3f}s - {end:.3f}s")

        print(f"  ... total {len(alignment.get('characters', []))} characters")


def test_tts_with_voice_settings(model: str, voice: str, api_key: str | None = None):
    """Test TTS with custom voice settings."""
    print("\n" + "=" * 60)
    print("TEST 3: TTS with Voice Settings")
    print("=" * 60)

    response = litellm.speech(
        model=model,
        input="This audio has custom voice settings applied.",
        voice=voice,
        api_key=api_key,
        speed=1.2,  # Slightly faster
        extra_body={
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            }
        },
    )

    output_path = Path("test_output_voice_settings.mp3")
    with open(output_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

    print(f"✓ Audio saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    print("  Voice settings: stability=0.5, similarity_boost=0.75, speed=1.2")


def test_tts_with_timestamps_and_settings(
    model: str, voice: str, api_key: str | None = None
):
    """Test TTS with both timestamps and custom voice settings."""
    print("\n" + "=" * 60)
    print("TEST 4: TTS with Timestamps + Voice Settings")
    print("=" * 60)

    response = litellm.speech(
        model=model,
        input="Combined test with timestamps and voice settings.",
        voice=voice,
        api_key=api_key,
        extra_body={
            "with_timestamps": True,
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.8,
                "speed": 0.9,
            },
        },
    )

    assert isinstance(response, dict), f"Expected dict, got {type(response)}"
    print("✓ Response is a dict (as expected)")

    if "audio_base64" in response:
        audio_bytes = base64.b64decode(response["audio_base64"])
        output_path = Path("test_output_combined.mp3")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"✓ Audio saved to: {output_path}")

    if "alignment" in response:
        print(f"✓ Alignment data present: {len(response['alignment'].get('characters', []))} characters")


def test_tts_with_pronunciation_dictionary(
    model: str,
    voice: str,
    api_key: str | None = None,
    pron_dict_id: str | None = None,
    pron_dict_version: str | None = None,
):
    """Test TTS with pronunciation dictionary (requires dict_id and version_id)."""
    print("\n" + "=" * 60)
    print("TEST 5: TTS with Pronunciation Dictionary")
    print("=" * 60)

    if not pron_dict_id or not pron_dict_version:
        print("⚠ Skipped: --pron-dict-id and --pron-dict-version required")
        return

    response = litellm.speech(
        model=model,
        input="Testing pronunciation dictionary with litellm.",
        voice=voice,
        api_key=api_key,
        extra_body={
            "with_timestamps": True,
            "pronunciation_dictionary_locators": [
                {
                    "pronunciation_dictionary_id": pron_dict_id,
                    "version_id": pron_dict_version,
                }
            ],
            "apply_text_normalization": "auto",
        },
    )

    assert isinstance(response, dict), f"Expected dict, got {type(response)}"
    print("✓ Response is a dict (as expected)")
    print(f"  Dictionary ID: {pron_dict_id}")
    print(f"  Version ID: {pron_dict_version}")

    if "audio_base64" in response:
        audio_bytes = base64.b64decode(response["audio_base64"])
        output_path = Path("test_output_pronunciation.mp3")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"✓ Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test ElevenLabs TTS with timestamps")
    parser.add_argument(
        "--proxy",
        type=str,
        help="LiteLLM proxy URL (e.g., http://localhost:4000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="elevenlabs/eleven_turbo_v2",
        help="Model to use (default: elevenlabs/eleven_turbo_v2)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="alloy",
        help="Voice to use (default: alloy, maps to Rachel)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)",
    )
    parser.add_argument(
        "--test",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific test only (1=regular, 2=timestamps, 3=voice_settings, 4=combined, 5=pronunciation_dict)",
    )
    parser.add_argument(
        "--pron-dict-id",
        type=str,
        help="Pronunciation dictionary ID (for test 5)",
    )
    parser.add_argument(
        "--pron-dict-version",
        type=str,
        help="Pronunciation dictionary version ID (for test 5)",
    )
    args = parser.parse_args()

    # Configure proxy if specified
    if args.proxy:
        litellm.api_base = args.proxy
        print(f"Using proxy: {args.proxy}")

    api_key = args.api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key and not args.proxy:
        print("ERROR: Set ELEVENLABS_API_KEY or use --api-key or --proxy")
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Voice: {args.voice}")

    # Store pronunciation dict args for test 5
    pron_dict_id = args.pron_dict_id
    pron_dict_version = args.pron_dict_version

    tests = {
        1: ("Regular TTS", lambda m, v, k: test_regular_tts(m, v, k)),
        2: ("TTS with Timestamps", lambda m, v, k: test_tts_with_timestamps(m, v, k)),
        3: ("TTS with Voice Settings", lambda m, v, k: test_tts_with_voice_settings(m, v, k)),
        4: ("TTS with Timestamps + Settings", lambda m, v, k: test_tts_with_timestamps_and_settings(m, v, k)),
        5: ("TTS with Pronunciation Dictionary", lambda m, v, k: test_tts_with_pronunciation_dictionary(m, v, k, pron_dict_id, pron_dict_version)),
    }

    if args.test:
        name, func = tests[args.test]
        func(args.model, args.voice, api_key)
    else:
        for num, (name, func) in tests.items():
            try:
                func(args.model, args.voice, api_key)
            except Exception as e:
                print(f"\n✗ {name} FAILED: {e}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
