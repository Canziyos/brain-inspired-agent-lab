import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from src.core.actions import Action
from src.telemetry.metrics import StepMetrics

ARBITRATION_READINESS_FIELDS: Final[tuple[str, ...]] = (
    "channel",
    "step_count",
    "usable_count",
    "usable_agreement_count",
    "usable_disagreement_count",
    "usable_disagreement_rate",
    "rescue_candidate_count",
    "override_risk_count",
    "neutral_disagreement_count",
    "rescue_candidate_rate",
    "override_risk_rate",
    "readiness_score",
    "mean_rule_reward_when_usable_agreed",
    "mean_rule_reward_when_usable_disagreed",
    "mean_expected_reward_when_usable_disagreed",
    "mean_confidence_when_usable_disagreed",
    "mean_reliability_when_usable_disagreed",
)

ActionGetter = Callable[[StepMetrics], Action | None]
UsableGetter = Callable[[StepMetrics], bool]
FloatGetter = Callable[[StepMetrics], float]



def write_arbitration_readiness_csv(
    history: Sequence[StepMetrics],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = summarize_arbitration_readiness(history)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=ARBITRATION_READINESS_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)



def summarize_arbitration_readiness(
    history: Sequence[StepMetrics],
) -> tuple[dict[str, Any], ...]:
    return (
        summarize_channel(
            channel="same_run_episodic",
            history=history,
            action_getter=lambda item: item.episodic_same_run_action,
            usable_getter=lambda item: item.episodic_same_run_is_usable,
            expected_getter=lambda item: item.episodic_same_run_expected_reward,
            confidence_getter=lambda item: item.episodic_same_run_confidence,
            reliability_getter=lambda item: item.episodic_same_run_reliability,
        ),
        summarize_channel(
            channel="prior_episodic",
            history=history,
            action_getter=lambda item: item.episodic_prior_action,
            usable_getter=lambda item: item.episodic_prior_is_usable,
            expected_getter=lambda item: item.episodic_prior_expected_reward,
            confidence_getter=lambda item: item.episodic_prior_confidence,
            reliability_getter=lambda item: item.episodic_prior_reliability,
        ),
        summarize_channel(
            channel="combined_episodic",
            history=history,
            action_getter=lambda item: item.episodic_action,
            usable_getter=lambda item: item.episodic_is_usable,
            expected_getter=lambda item: item.episodic_expected_reward,
            confidence_getter=lambda item: item.episodic_confidence,
            reliability_getter=lambda item: item.episodic_reliability,
        ),
        summarize_channel(
            channel="neural_imagination",
            history=history,
            action_getter=lambda item: item.imagination_action,
            usable_getter=lambda item: True,
            expected_getter=lambda item: item.imagination_expected_reward,
            confidence_getter=lambda item: 0.0,
            reliability_getter=lambda item: 0.0,
        ),
    )



def summarize_channel(
    channel: str,
    history: Sequence[StepMetrics],
    action_getter: ActionGetter,
    usable_getter: UsableGetter,
    expected_getter: FloatGetter,
    confidence_getter: FloatGetter,
    reliability_getter: FloatGetter,
) -> dict[str, Any]:
    step_count = len(history)
    usable_items = tuple(
        item
        for item in history
        if action_getter(item) is not None and usable_getter(item)
    )
    usable_agreement_items = tuple(
        item
        for item in usable_items
        if action_getter(item) is item.rule_action
    )
    usable_disagreement_items = tuple(
        item
        for item in usable_items
        if action_getter(item) is not item.rule_action
    )

    rescue_candidate_items = tuple(
        item
        for item in usable_disagreement_items
        if item.reward < 0.0
    )
    override_risk_items = tuple(
        item
        for item in usable_disagreement_items
        if item.reward > 0.0
    )
    neutral_disagreement_count = (
        len(usable_disagreement_items)
        - len(rescue_candidate_items)
        - len(override_risk_items)
    )

    rescue_rate = rate(
        len(rescue_candidate_items),
        len(usable_disagreement_items),
    )
    override_risk_rate = rate(
        len(override_risk_items),
        len(usable_disagreement_items),
    )

    return {
        "channel": channel,
        "step_count": step_count,
        "usable_count": len(usable_items),
        "usable_agreement_count": len(usable_agreement_items),
        "usable_disagreement_count": len(usable_disagreement_items),
        "usable_disagreement_rate": rate(
            len(usable_disagreement_items),
            len(usable_items),
        ),
        "rescue_candidate_count": len(rescue_candidate_items),
        "override_risk_count": len(override_risk_items),
        "neutral_disagreement_count": neutral_disagreement_count,
        "rescue_candidate_rate": rescue_rate,
        "override_risk_rate": override_risk_rate,
        "readiness_score": rescue_rate - override_risk_rate,
        "mean_rule_reward_when_usable_agreed": mean(
            item.reward
            for item in usable_agreement_items
        ),
        "mean_rule_reward_when_usable_disagreed": mean(
            item.reward
            for item in usable_disagreement_items
        ),
        "mean_expected_reward_when_usable_disagreed": mean(
            expected_getter(item)
            for item in usable_disagreement_items
        ),
        "mean_confidence_when_usable_disagreed": mean(
            confidence_getter(item)
            for item in usable_disagreement_items
        ),
        "mean_reliability_when_usable_disagreed": mean(
            reliability_getter(item)
            for item in usable_disagreement_items
        ),
    }



def rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator



def mean(values: Sequence[float] | Any) -> float:
    items = tuple(values)
    if not items:
        return 0.0

    return sum(items) / len(items)
