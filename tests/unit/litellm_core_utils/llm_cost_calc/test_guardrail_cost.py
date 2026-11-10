import datetime
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
    bedrock_guardrail_cost,
    bedrock_guardrail_cost_by_unit,
    bedrock_inline_guardrail_cost,
    bedrock_inline_guardrail_usage_units,
    billed_guardrail_cost_by_unit,
    cost_breakdown_with_guardrail,
    guardrail_cost_total,
    guardrail_information_cost,
)
from litellm.types.utils import Choices, Message, ModelResponse, Usage


@pytest.fixture
def synthetic_cost_map(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "bedrock/guardrails": {
                "guardrail_cost_per_unit": {
                    "contentPolicyUnits": 0.00015,
                    "topicPolicyUnits": 0.00015,
                    "wordPolicyUnits": 0.0,
                }
            },
            "bedrock/eu-west-1/guardrails": {"guardrail_cost_per_unit": {"contentPolicyUnits": 0.0002}},
            "bedrock/us-west-2/guardrails": {"guardrail_cost_per_unit": "malformed"},
        },
    )


def test_bedrock_guardrail_cost_prices_each_counter(synthetic_cost_map):
    cost = bedrock_guardrail_cost(
        usage_units={"contentPolicyUnits": 2, "topicPolicyUnits": 1, "wordPolicyUnits": 5},
        aws_region_name="us-east-1",
    )
    assert cost == pytest.approx(0.00045)


def test_bedrock_guardrail_cost_prefers_regional_entry(synthetic_cost_map):
    cost = bedrock_guardrail_cost(usage_units={"contentPolicyUnits": 1}, aws_region_name="eu-west-1")
    assert cost == pytest.approx(0.0002)


def test_bedrock_guardrail_cost_unknown_counter_is_free(synthetic_cost_map):
    assert bedrock_guardrail_cost(usage_units={"someFutureCounter": 3}, aws_region_name="us-east-1") == 0.0


def test_bedrock_guardrail_cost_malformed_regional_entry_falls_back(synthetic_cost_map):
    cost = bedrock_guardrail_cost(usage_units={"contentPolicyUnits": 1}, aws_region_name="us-west-2")
    assert cost == pytest.approx(0.00015)


def test_bedrock_guardrail_cost_no_pricing_entry(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", {})
    assert bedrock_guardrail_cost(usage_units={"contentPolicyUnits": 1}, aws_region_name="us-east-1") == 0.0


def test_bedrock_guardrail_cost_by_unit_prices_every_counter_it_was_given(synthetic_cost_map):
    """LIT-5652: the daily rollup stores one row per counter, so pricing must come
    back at that grain, keyed exactly like the usage. An explicit 0.0 in the cost
    map is free; a counter the map does not list is unknown (None), never free,
    while the scalar the spend path bills still sums only the known prices."""
    usage = {"contentPolicyUnits": 2, "topicPolicyUnits": 1, "wordPolicyUnits": 5, "someFutureCounter": 3}
    by_unit = bedrock_guardrail_cost_by_unit(usage_units=usage, aws_region_name="us-east-1")
    assert by_unit is not None
    assert by_unit.keys() == usage.keys()
    assert by_unit["contentPolicyUnits"] == pytest.approx(0.0003)
    assert by_unit["topicPolicyUnits"] == pytest.approx(0.00015)
    assert by_unit["wordPolicyUnits"] == 0.0
    assert by_unit["someFutureCounter"] is None
    assert guardrail_cost_total(by_unit) == pytest.approx(0.00045)
    assert guardrail_cost_total(by_unit) == pytest.approx(
        bedrock_guardrail_cost(usage_units=usage, aws_region_name="us-east-1")
    )


def test_bedrock_guardrail_cost_by_unit_is_none_without_pricing_so_unpriced_is_not_free(monkeypatch):
    """The scalar keeps returning 0.0 for the spend path; the per-unit view must
    say "unknown" instead so the rollup stores NULL rather than a $0 that would
    hide the exact silent-spend problem this feature exists to surface."""
    monkeypatch.setattr(litellm, "model_cost", {})
    assert bedrock_guardrail_cost_by_unit(usage_units={"contentPolicyUnits": 1}, aws_region_name="us-east-1") is None


def test_billed_guardrail_cost_by_unit_reads_the_hook_stamp():
    entry = {
        "guardrail_name": "bedrock",
        "guardrail_cost_by_unit": {"contentPolicyUnits": 0.15, "wordPolicyUnits": 0, "someFutureCounter": None},
    }
    assert billed_guardrail_cost_by_unit(entry) == {
        "contentPolicyUnits": 0.15,
        "wordPolicyUnits": 0.0,
        "someFutureCounter": None,
    }


@pytest.mark.parametrize(
    "entry",
    [
        {"guardrail_name": "no-pricing", "guardrail_usage": {"contentPolicyUnits": 1}},
        {"guardrail_cost_by_unit": {"text_records": 0.5}, "guardrail_cost_in_spend": False},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": -0.5}},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": float("nan")}},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": float("inf")}},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": "bad"}},
        {"guardrail_cost_by_unit": "not-a-map"},
        {"guardrail_cost_by_unit": {"contentPolicyUnits": 0.1}, "guardrail_cost_in_spend": "maybe"},
        "not-an-entry",
    ],
)
def test_billed_guardrail_cost_by_unit_is_none_when_unpriced_report_only_or_forged(entry):
    assert billed_guardrail_cost_by_unit(entry) is None


def test_billed_guardrail_cost_by_unit_treats_none_in_spend_as_billed():
    entry = {"guardrail_cost_by_unit": {"contentPolicyUnits": 0.15}, "guardrail_cost_in_spend": None}
    assert billed_guardrail_cost_by_unit(entry) == {"contentPolicyUnits": 0.15}


def test_guardrail_information_cost_sums_entries():
    entries = [
        {"guardrail_name": "a", "guardrail_cost": 0.0003},
        {"guardrail_name": "b", "guardrail_cost": None},
        {"guardrail_name": "c"},
        {"guardrail_name": "d", "guardrail_cost": 0.0001},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.0004)


def test_guardrail_information_cost_single_entry_and_garbage():
    assert guardrail_information_cost({"guardrail_cost": 0.0001}) == pytest.approx(0.0001)
    assert guardrail_information_cost(None) == 0.0
    assert guardrail_information_cost("not-guardrail-info") == 0.0
    assert guardrail_information_cost([{"guardrail_cost": "bad"}]) == 0.0


def test_guardrail_information_cost_ignores_negative_and_non_finite():
    entries = [
        {"guardrail_name": "forged-negative", "guardrail_cost": -0.005},
        {"guardrail_name": "forged-nan", "guardrail_cost": float("nan")},
        {"guardrail_name": "forged-inf", "guardrail_cost": float("inf")},
        {"guardrail_name": "real", "guardrail_cost": 0.0003},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.0003)
    assert guardrail_information_cost({"guardrail_cost": -1.0}) == 0.0


def test_cost_breakdown_with_guardrail_merges_and_creates():
    assert cost_breakdown_with_guardrail(None, 0.0) is None
    untouched = {"input_cost": 0.1, "total_cost": 0.4}
    assert cost_breakdown_with_guardrail(untouched, 0.0) is untouched
    merged = cost_breakdown_with_guardrail({"input_cost": 0.1, "total_cost": 0.4}, 0.0003)
    assert merged is not None
    assert merged["guardrail_cost"] == pytest.approx(0.0003)
    assert merged["total_cost"] == pytest.approx(0.4003)
    assert merged["input_cost"] == pytest.approx(0.1)
    created = cost_breakdown_with_guardrail(None, 0.0003)
    assert created == {"guardrail_cost": 0.0003, "total_cost": 0.0003}


def test_azure_prompt_shield_guardrail_cost_paid_tier_prices_text_records():
    from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
        azure_prompt_shield_guardrail_cost,
    )

    cost = azure_prompt_shield_guardrail_cost(
        usage_units={"text_records": 3, "requests": 1, "input_characters": 2100},
        cost_tier="paid",
        price_per_1000_text_records=0.38,
    )
    assert cost == pytest.approx(0.00114)


def test_azure_prompt_shield_guardrail_cost_free_tier_is_zero():
    from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
        azure_prompt_shield_guardrail_cost,
    )

    assert azure_prompt_shield_guardrail_cost({"text_records": 50}, "free", 0.38) == 0.0


def test_azure_prompt_shield_guardrail_cost_unconfigured_is_none():
    from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
        azure_prompt_shield_guardrail_cost,
    )

    assert azure_prompt_shield_guardrail_cost({"text_records": 50}, None, None) is None


def test_azure_prompt_shield_guardrail_cost_no_text_records_is_zero():
    from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
        azure_prompt_shield_guardrail_cost,
    )

    assert azure_prompt_shield_guardrail_cost({}, None, 0.38) == 0.0


def test_guardrail_information_cost_excludes_entries_marked_not_in_spend():
    entries = [
        {"guardrail_name": "azure-shield", "guardrail_cost": 0.5, "guardrail_cost_in_spend": False},
        {"guardrail_name": "bedrock", "guardrail_cost": 0.0003},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.0003)
    assert guardrail_information_cost({"guardrail_cost": 0.5, "guardrail_cost_in_spend": False}) == 0.0
    assert guardrail_information_cost({"guardrail_cost": 0.5, "guardrail_cost_in_spend": True}) == pytest.approx(0.5)


def test_guardrail_information_cost_treats_none_in_spend_as_billed():
    """An explicit ``guardrail_cost_in_spend: None`` (the TypedDict sanctions it)
    keeps the default billed behavior AND must not fail union validation, which
    would silently zero a sibling entry's real cost."""
    assert guardrail_information_cost({"guardrail_cost": 0.5, "guardrail_cost_in_spend": None}) == pytest.approx(0.5)
    entries = [
        {"guardrail_name": "azure-shield", "guardrail_cost": 0.5, "guardrail_cost_in_spend": None},
        {"guardrail_name": "bedrock", "guardrail_cost": 0.0003},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.5003)


def test_guardrail_information_cost_skips_malformed_entry_keeps_siblings():
    """Entries are validated one by one: a malformed entry (a custom hook stamping
    a non-boolean guardrail_cost_in_spend) prices to 0.0 by itself and must not
    zero a sibling entry's real billable cost."""
    entries = [
        {"guardrail_name": "custom", "guardrail_cost": 0.5, "guardrail_cost_in_spend": "maybe"},
        {"guardrail_name": "bedrock", "guardrail_cost": 0.0003},
    ]
    assert guardrail_information_cost(entries) == pytest.approx(0.0003)
    assert guardrail_information_cost({"guardrail_cost": 0.5, "guardrail_cost_in_spend": "maybe"}) == 0.0


def _converse_guardrail_usage(topic_units: int, content_units: int) -> dict:
    return {
        "invocationMetrics": {
            "guardrailProcessingLatency": 215,
            "usage": {
                "topicPolicyUnits": topic_units,
                "contentPolicyUnits": content_units,
                "wordPolicyUnits": 0,
                "sensitiveInformationPolicyUnits": 0,
                "sensitiveInformationPolicyFreeUnits": 0,
                "contextualGroundingPolicyUnits": 0,
            },
            "guardrailCoverage": {"textCharacters": {"guarded": 62, "total": 62}},
        }
    }


# Shape captured from a real Converse call with guardrailConfig trace=enabled_full
_CONVERSE_GUARDRAIL_TRACE: Final = {
    "guardrail": {
        "modelOutput": ['{"content":[{"text":"Hello!"}]}'],
        "inputAssessment": {"kw7zahiis45j": _converse_guardrail_usage(topic_units=1, content_units=2)},
        "outputAssessments": {
            "kw7zahiis45j": [
                _converse_guardrail_usage(topic_units=1, content_units=1),
                _converse_guardrail_usage(topic_units=3, content_units=0),
            ]
        },
    }
}


def test_bedrock_inline_guardrail_usage_units_sums_input_and_every_output_assessment():
    units = bedrock_inline_guardrail_usage_units(_CONVERSE_GUARDRAIL_TRACE)

    assert units["topicPolicyUnits"] == 5
    assert units["contentPolicyUnits"] == 3
    assert units["wordPolicyUnits"] == 0


@pytest.mark.parametrize(
    "trace",
    [None, {}, {"guardrail": {}}, {"guardrail": "malformed"}, {"guardrail": {"inputAssessment": {"g": {}}}}],
)
def test_bedrock_inline_guardrail_usage_units_empty_without_usage(trace):
    assert not bedrock_inline_guardrail_usage_units(trace)


def test_bedrock_inline_guardrail_cost_prices_trace_on_response(synthetic_cost_map):
    response = SimpleNamespace(trace=_CONVERSE_GUARDRAIL_TRACE)

    assert bedrock_inline_guardrail_cost(response, aws_region_name="us-east-1") == pytest.approx(8 * 0.00015)


def test_bedrock_inline_guardrail_cost_is_zero_without_trace(synthetic_cost_map):
    assert bedrock_inline_guardrail_cost(SimpleNamespace(), aws_region_name="us-east-1") == 0.0


def _bedrock_model_response(trace: dict | None) -> ModelResponse:
    response = ModelResponse(
        model="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        choices=[Choices(index=0, finish_reason="stop", message=Message(role="assistant", content="Hello!"))],
        usage=Usage(prompt_tokens=19, completion_tokens=10, total_tokens=29),
    )
    if trace is not None:
        setattr(response, "trace", trace)
    return response


def _logging_obj() -> Logging:
    return Logging(
        model="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime.datetime(2026, 11, 10, 12, 0, 0),
        litellm_call_id="inline-guardrail-cost",
        function_id="f",
    )


def test_completion_cost_reports_bedrock_inline_guardrail_cost_outside_total():
    model: Final = "bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0"
    token_only_cost: Final = litellm.completion_cost(completion_response=_bedrock_model_response(None), model=model)
    logging_obj: Final = _logging_obj()

    with_guardrail_cost: Final = litellm.completion_cost(
        completion_response=_bedrock_model_response(_CONVERSE_GUARDRAIL_TRACE),
        model=model,
        litellm_logging_obj=logging_obj,
    )

    expected_guardrail_cost: Final = sum(
        units * litellm.model_cost["bedrock/guardrails"]["guardrail_cost_per_unit"][counter]
        for counter, units in bedrock_inline_guardrail_usage_units(_CONVERSE_GUARDRAIL_TRACE).items()
    )
    assert expected_guardrail_cost > 0
    assert with_guardrail_cost == pytest.approx(token_only_cost)
    assert logging_obj.cost_breakdown["total_cost"] == pytest.approx(token_only_cost)
    assert logging_obj.cost_breakdown["bedrock_inline_guardrail_cost"] == pytest.approx(expected_guardrail_cost)


def test_completion_cost_omits_bedrock_inline_guardrail_cost_without_trace():
    logging_obj: Final = _logging_obj()

    litellm.completion_cost(
        completion_response=_bedrock_model_response(None),
        model="bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        litellm_logging_obj=logging_obj,
    )

    assert "bedrock_inline_guardrail_cost" not in logging_obj.cost_breakdown
