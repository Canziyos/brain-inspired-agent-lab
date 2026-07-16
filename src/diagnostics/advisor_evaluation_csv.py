import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from src.core.actions import Action
from src.simulation.step_action_text import format_action
from src.telemetry.metrics import StepMetrics

ADVISOR_EVALUATION_FIELDS: Final[tuple[str, ...]] = (
    "channel",
    "step_count",
    "advice_count",
    "usable_count",
    "agreement_count",
    "usable_agreement_count",
    "advice_rate",
    "usable_rate",
    "agreement_rate",
    "usable_agreement_rate",
    "mean_rule_reward_all_steps",
    "mean_rule_reward_when_advice_agreed",
    "mean_rule_reward_when_advice_disagreed",
    "mean_rule_reward_when_usable_agreed",
    "mean_rule_reward_when_usable_disagreed",
    "mean_expected_reward",
    "mean_confidence",
    "mean_reliability",
)

ActionGetter = Callable[[StepMetrics], Action | None]
UsableGetter = Callable[[StepMetrics], bool]
FloatGetter = Callable[[StepMetrics], float]



def write_advisor_evaluation_csv(
    history: Sequence[StepMetrics],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = summarize_advisor_evaluation(history)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=ADVISOR_EVALUATION_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)



def summarize_advisor_evaluation(
    history: Sequence[StepMetrics],
) -> tuple[dict[str, Any], ...]:
    return (
        summarize_rule_policy(history),
        summarize_channel(
            channel="same_run_episodic",
            history=history,
            action_getter=lambda item: item.episodic_same_run_action,
            usable_getter=lambda item: item.episodic_same_run_is_usable,
            expected_getter=(
                lambda item: item.episodic_same_run_expected_reward
            ),
            confidence_getter=(
                lambda item: item.episodic_same_run_confidence
            ),
            reliability_getter=(
                lambda item: item.episodic_same_run_reliability
            ),
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



def summarize_rule_policy(
    history: Sequence[StepMetrics],
) -> dict[str, Any]:
    step_count = len(history)
    mean_reward = mean(
        item.reward
        for item in history
    )

    return {
        "channel": "rule_policy",
        "step_count": step_count,
        "advice_count": step_count,
        "usable_count": step_count,
        "agreement_count": step_count,
        "usable_agreement_count": step_count,
        "advice_rate": rate(step_count, step_count),
        "usable_rate": rate(step_count, step_count),
        "agreement_rate": rate(step_count, step_count),
        "usable_agreement_rate": rate(step_count, step_count),
        "mean_rule_reward_all_steps": mean_reward,
        "mean_rule_reward_when_advice_agreed": mean_reward,
        "mean_rule_reward_when_advice_disagreed": 0.0,
        "mean_rule_reward_when_usable_agreed": mean_reward,
        "mean_rule_reward_when_usable_disagreed": 0.0,
        "mean_expected_reward": 0.0,
        "mean_confidence": 0.0,
        "mean_reliability": 0.0,
    }



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
    advice_items = tuple(
        item
        for item in history
        if action_getter(item) is not None
    )
    usable_items = tuple(
        item
        for item in advice_items
        if usable_getter(item)
    )
    agreement_items = tuple(
        item
        for item in advice_items
        if action_getter(item) is item.rule_action
    )
    disagreement_items = tuple(
        item
        for item in advice_items
        if action_getter(item) is not item.rule_action
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

    return {
        "channel": channel,
        "step_count": step_count,
        "advice_count": len(advice_items),
        "usable_count": len(usable_items),
        "agreement_count": len(agreement_items),
        "usable_agreement_count": len(usable_agreement_items),
        "advice_rate": rate(len(advice_items), step_count),
        "usable_rate": rate(len(usable_items), step_count),
        "agreement_rate": rate(len(agreement_items), len(advice_items)),
        "usable_agreement_rate": rate(
            len(usable_agreement_items),
            len(usable_items),
        ),
        "mean_rule_reward_all_steps": mean(
            item.reward
            for item in history
        ),
        "mean_rule_reward_when_advice_agreed": mean(
            item.reward
            for item in agreement_items
        ),
        "mean_rule_reward_when_advice_disagreed": mean(
            item.reward
            for item in disagreement_items
        ),
        "mean_rule_reward_when_usable_agreed": mean(
            item.reward
            for item in usable_agreement_items
        ),
        "mean_rule_reward_when_usable_disagreed": mean(
            item.reward
            for item in usable_disagreement_items
        ),
        "mean_expected_reward": mean(
            expected_getter(item)
            for item in advice_items
        ),
        "mean_confidence": mean(
            confidence_getter(item)
            for item in advice_items
        ),
        "mean_reliability": mean(
            reliability_getter(item)
            for item in advice_items
        ),
    }



def format_action_or_empty(action: Action | None) -> str:
    if action is None:
        return ""

    return format_action(action)



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
