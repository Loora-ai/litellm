# Loora LiteLLM fork — drop this into chat on the next upgrade

This file is the handoff for the next LiteLLM bump. Attach it instead of re-researching the fork.

## Current branch

- Branch: `1-98-0` (same convention as `1-87-4`, `1-87-4-speechace`)
- Upstream tag: `v1.98.0` (`BerriAI/litellm`)
- Fork remote: `origin` = `Loora-ai/litellm`

Previous production branch: `1-87-4-speechace` on `v1.87.4`.

## What we keep (Loora-only)

Do **not** cherry-pick the old `"changes"` commits. Re-apply these features on the new tag.

### 1. SpeechAce — `POST /v1/speechace/score`

Dedicated connector. Not a generic pass-through YAML route.

- `litellm/proxy/pass_through_endpoints/speechace_connector.py`
- `litellm/proxy/_types.py` — `speechace_routes` on `LiteLLMRoutes.llm_api_routes`
- `litellm/proxy/proxy_server.py` — `app.include_router(speechace_connector_router)`
- Tests: `tests/test_litellm/proxy/pass_through_endpoints/test_speechace_connector.py`

Env:

- `SPEECHACE_TARGET_URL`
- `SPEECHACE_API_KEY`
- `SPEECHACE_USER_ID`

Auth must **not** read the multipart body (audio is streamed upstream). Logging/metrics run after the response. `custom_llm_provider="speechace"` is set for Prometheus. `x-litellm-metadata` is merged in the connector (not via generic pass-through).

**Upgrade risk:** this file calls private auth helpers (`_user_api_key_auth_builder`, `_run_centralized_common_checks`, `_init_kwargs_for_pass_through_endpoint`). Re-check signatures; do not copy blindly.

### 2. ElevenLabs TTS — `with_timestamps` + `voice_settings`

Upstream TTS is still binary-only. We add `/with-timestamps` (JSON `audio_base64` + alignment) and merge `voice_settings`.

- `litellm/llms/elevenlabs/text_to_speech/transformation.py`
- `litellm/llms/base_llm/text_to_speech/transformation.py` — response may be `dict`
- `litellm/llms/custom_httpx/llm_http_handler.py` — TTS handler return type
- `litellm/main.py` — `speech()` copies timestamps key into `litellm_params`; return type allows `dict`
- `litellm/proxy/proxy_server.py` — `/v1/audio/speech` returns `ORJSONResponse` when the result is a dict
- `litellm/types/llms/elevenlabs.py`
- Tests: `tests/test_litellm/llms/elevenlabs/test_elevenlabs_text_to_speech_transformation.py`
- Optional script: `scripts/test_elevenlabs_tts.py`

### 3. Prometheus

Two layers. Both are required.

**A. Provider sync (empty `api_provider` without this)**

TTS/`speech()` and SpeechAce pass `custom_llm_provider` as a kwarg. Deployment metrics read `litellm_params["custom_llm_provider"]`.

- `litellm/litellm_core_utils/litellm_logging.py` — `update_environment_variables` copies it into `litellm_params`
- `litellm/main.py` — put `custom_llm_provider` **inside** `litellm_params` for `speech()` and `transcription()`
- Tests: `test_update_environment_variables_syncs_custom_llm_provider_into_litellm_params`, `tests/test_litellm/integrations/test_prometheus_tts_api_provider.py`

**B. Extra labels (`api_base` on request metrics)**

Defined in `litellm/types/integrations/prometheus.py` (`PrometheusMetricLabels`). Fallback emit path fills values in `litellm/integrations/prometheus.py` (`log_success_fallback_event` / `log_failure_fallback_event`).

v1.98.0 already has `api_provider` on most request metrics, and **already has `api_base` on deployment metrics** (`litellm_deployment_state`, `litellm_deployment_total_requests`, `litellm_deployment_success_responses`, `litellm_deployment_failure_responses`, `litellm_deployment_cooled_down`, `litellm_deployment_latency_per_output_token`, TPM/RPM limits).

We add **`api_base`** only where it was missing (request/spend/token/latency/proxy/cache/batch metrics), plus `team`/`team_alias` on remaining requests/tokens, plus `api_provider`+`api_base` on fallback metrics.

**Do not append a label a metric already has.** prometheus-client 0.20 accepts duplicate `labelnames` at Gauge construction, then `.labels(**kwargs)` raises `ValueError: Incorrect label names` (kwargs are unique keys). That broke `set_llm_deployment_success_metrics` in production.

Adding labels creates new Prometheus series. Old series without `api_base` stop updating. Grafana that filters by `api_base` needs this.

## What we do **not** keep

- Generic pass-through YAML extras (`os.environ/` on `target` / `default_query_params`, YAML `custom_llm_provider`, multipart query-merge fix). SpeechAce does not use them.
- Blind cherry-pick of `1-87-4-speechace` onto a new tag. `pass_through_endpoints.py` and TTS were rewritten.

## How to bump next time

```text
1. git fetch upstream --tags
2. Branch 1-XX-Y from v1.XX.Y  (example: 1-99-0 from v1.99.0)
3. Re-apply the three features above on the new code. Do not replay old commits.
4. Run:
   - tests/test_litellm/proxy/pass_through_endpoints/test_speechace_connector.py
   - tests/test_litellm/llms/elevenlabs/test_elevenlabs_text_to_speech_transformation.py
   - the prometheus provider-sync tests
5. Diff PrometheusMetricLabels vs the new tag. Only add labels we still need; do not duplicate `api_base` / `api_provider` if upstream already has them. Run `test_prometheus_metric_label_lists_have_no_duplicates` and `test_set_deployment_healthy_accepts_standard_labels`.
```

## Deploy config (outside this repo)

- `loora-k8s-deployments` — SpeechAce env + ElevenLabs model list
- Prometheus: `custom_prometheus_metadata_labels: [metadata.service, metadata.module, metadata.prompt]`
- Alerts: `loora-k8s-deployments/.../alertmanager-rules/rules/litellm-alerts.yaml`
