# Loora LiteLLM fork — drop this into chat on the next upgrade

This file is the handoff for the next LiteLLM bump. Attach it instead of re-researching the fork.

## Current branch

- Branch: `1-104-0` (same convention as `1-98-0`, `1-87-4`, `1-87-4-speechace`)
- Upstream tag: `v1.104.0` (`BerriAI/litellm`)
- Fork remote: `origin` = `Loora-ai/litellm`

Previous production branch: `1-98-0` on `v1.98.0`.

## What we keep (Loora-only)

Do **not** cherry-pick the old `"changes"` commits. Re-apply these features on the new tag. Upstream rewrites the touched files between releases, so a patch from `1-104-0` will not apply cleanly onto a later tag.

### 1. SpeechAce — `POST /v1/speechace/score`

Dedicated connector. Not a generic pass-through YAML route.

- `litellm/proxy/pass_through_endpoints/speechace_connector.py` (new file; copy it forward)
- `litellm/proxy/_types.py` — `speechace_routes = ["/v1/speechace/score"]` included in `LiteLLMRoutes.llm_api_routes`
- `litellm/proxy/proxy_server.py` — import `speechace_connector_router` and `app.include_router(speechace_connector_router)` next to the generic pass-through router
- Tests: `tests/test_litellm/proxy/pass_through_endpoints/test_speechace_connector.py`

Env:

- `SPEECHACE_TARGET_URL`
- `SPEECHACE_API_KEY`
- `SPEECHACE_USER_ID`

Auth must **not** read the multipart body (audio is streamed upstream). Logging/metrics run after the response. `custom_llm_provider="speechace"` is set on `litellm_params` for Prometheus. `x-litellm-metadata` is merged in the connector (not via generic pass-through).

**Upgrade risk:** this file calls private auth helpers (`_user_api_key_auth_builder`, `_run_centralized_common_checks`, `_ensure_parent_otel_span_on_request_state`) and `HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint` / `get_merged_query_parameters` / `get_response_headers`. Re-check signatures before copying the file. On `v1.104.0` the signatures were still compatible (`dict` vs `dict[str, object]` only).

### 2. ElevenLabs TTS — `with_timestamps` + `voice_settings`

Upstream TTS is still binary-only. We add `/with-timestamps` (JSON `audio_base64` + alignment) and merge `voice_settings` from params, kwargs, and `extra_body`. OpenAI `speed` is folded into `voice_settings.speed`.

- `litellm/llms/elevenlabs/text_to_speech/transformation.py`
  - `ELEVENLABS_WITH_TIMESTAMPS_KEY = "__elevenlabs_with_timestamps__"`
  - `get_complete_url` appends `/with-timestamps` when that key is set
  - `transform_text_to_speech_response` returns the JSON body when `content-type` contains `application/json`, otherwise binary audio
- `litellm/llms/base_llm/text_to_speech/transformation.py` — response may be `dict`
- `litellm/llms/custom_httpx/llm_http_handler.py` — TTS handler return type includes `dict`. **Do not** call `set_provider_response_headers_in_hidden_params` on a dict. `v1.104.0` added that call; a dict has no `_hidden_params` and it raises. Guard with `if not isinstance(speech_response, dict)`.
- `litellm/main.py` — `speech()` return type allows `dict`; copy the timestamps key from kwargs into `litellm_params` before the HTTP call
- `litellm/proxy/proxy_server.py` — `/v1/audio/speech` returns `ORJSONResponse` when the result is a dict (before `StreamingResponse`)
- `litellm/types/llms/elevenlabs.py` (new file; copy it forward)
- Tests: `tests/test_litellm/llms/elevenlabs/test_elevenlabs_text_to_speech_transformation.py`

### 3. Prometheus

Two layers. Both are required.

**A. Provider sync (empty `api_provider` without this)**

TTS/`speech()`, transcription, and SpeechAce pass `custom_llm_provider` as a kwarg. Deployment metrics read `litellm_params["custom_llm_provider"]`.

- `litellm/litellm_core_utils/litellm_logging.py` — `update_environment_variables` copies `additional_params["custom_llm_provider"]` into `self.litellm_params` **before** `model_call_details.update`
- `litellm/main.py` — put `custom_llm_provider` **inside** the `litellm_params` dict for `speech()` and `transcription()` (after `**kwargs`, so it wins)
- Tests: `test_update_environment_variables_syncs_custom_llm_provider_into_litellm_params` in `tests/unit/litellm_core_utils/test_litellm_logging.py`, and `tests/test_litellm/integrations/test_prometheus_tts_api_provider.py`

**B. Extra labels (`api_base` on request metrics)**

Defined in `litellm/types/integrations/prometheus.py` (`PrometheusMetricLabels`). The success-event path already fills `api_base` / `api_provider` on `UserAPIKeyLabelValues`, so adding the label name is enough for request/spend/token/latency/proxy/cache/batch metrics.

Fallback metrics do **not** fill those fields upstream. `log_success_fallback_event` and `log_failure_fallback_event` in `litellm/integrations/prometheus.py` must read `api_base` and `custom_llm_provider` from `standard_logging_object` or `litellm_params` and pass them into `UserAPIKeyLabelValues`.

`v1.104.0` already has `api_base` **and** `api_provider` on deployment metrics (`litellm_deployment_state`, `litellm_deployment_total_requests`, `litellm_deployment_success_responses`, `litellm_deployment_failure_responses`, `litellm_deployment_cooled_down`, `litellm_deployment_latency_per_output_token`, TPM/RPM limits, overhead). It also already has them on `litellm_remaining_requests_metric` / `litellm_remaining_tokens_metric`.

On `v1.104.0` we still added:

- `api_base` on request, spend, token, latency, queue time, proxy, cache, and managed-batch metrics that did not have it
- `team` / `team_alias` on `litellm_remaining_requests_metric` and `litellm_remaining_tokens_metric` (those gauges already receive team from the success-event enum)
- `api_provider` + `api_base` on `litellm_deployment_successful_fallbacks` (failed fallbacks alias that list)

**Do not append a label a metric already has.** prometheus-client 0.20 accepts duplicate `labelnames` at Gauge construction, then `.labels(**kwargs)` raises `ValueError: Incorrect label names` (kwargs are unique keys). That broke `set_llm_deployment_success_metrics` in production on `1.98.0`.

**Do not drop upstream labels while porting.** `v1.104.0` added `model_group` on several deployment metrics. Keep those. Only add labels that are still missing.

Adding labels creates new Prometheus series. Old series without `api_base` stop updating. Grafana that filters by `api_base` needs this.

## What we do **not** keep

- Generic pass-through YAML extras (`os.environ/` on `target` / `default_query_params`, YAML `custom_llm_provider`, multipart query-merge fix). SpeechAce does not use them.
- Blind cherry-pick of `1-87-4-speechace` or `1-98-0` onto a new tag. `pass_through_endpoints.py`, TTS, and `all_litellm_params` have been rewritten.
- The `client_side_timeout` entry that `1-98-0` added to the old `all_litellm_params` tuple. `v1.104.0` already strips that marker before the provider call (`OWNED_KWARG_NAMES`). Re-check on the next tag; do not add it again if upstream owns it.

## How to bump next time

```text
1. git fetch upstream tag v1.XX.Y --no-tags
   (a full `git fetch upstream --tags` is slow on this repo)
2. git checkout -b 1-XX-Y v1.XX.Y
3. Copy the Loora-only files from the previous branch (do not replay old commits):
   - litellm/proxy/pass_through_endpoints/speechace_connector.py
   - litellm/types/llms/elevenlabs.py
   - tests/test_litellm/proxy/pass_through_endpoints/test_speechace_connector.py
   - tests/test_litellm/llms/elevenlabs/test_elevenlabs_text_to_speech_transformation.py
   - tests/test_litellm/integrations/test_prometheus_tts_api_provider.py
4. Re-apply the edits in sections 1–3 against the new code. Diff
   PrometheusMetricLabels vs the new tag first. Only add labels we still
   need. Never duplicate api_base / api_provider. Never remove a label
   upstream added.
5. If text_to_speech_handler calls set_provider_response_headers_in_hidden_params,
   skip that call when the response is a dict.
6. Re-check SpeechAce private helper signatures.
7. Run:
   - tests/test_litellm/proxy/pass_through_endpoints/test_speechace_connector.py
   - tests/test_litellm/llms/elevenlabs/test_elevenlabs_text_to_speech_transformation.py
   - tests/test_litellm/integrations/test_prometheus_tts_api_provider.py
   - tests/unit/integrations/test_prometheus_labels.py
     (test_prometheus_metric_label_lists_have_no_duplicates,
      test_set_deployment_healthy_accepts_standard_labels)
   - tests/unit/litellm_core_utils/test_litellm_logging.py::test_update_environment_variables_syncs_custom_llm_provider_into_litellm_params
```

`v1.104.0` moved a lot of tests from `tests/test_litellm/` to `tests/unit/`. The Loora tests above still live under `tests/test_litellm/` and pytest collects both.

## Deploy config (outside this repo)

- `loora-k8s-deployments` — SpeechAce env + ElevenLabs model list. Point the image at the new fork tag after this branch is built.
- Prometheus: `custom_prometheus_metadata_labels: [metadata.service, metadata.module, metadata.prompt]`
- Alerts: `loora-k8s-deployments/.../alertmanager-rules/rules/litellm-alerts.yaml`
