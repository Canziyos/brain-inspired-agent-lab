from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from src.core.actions import Action
from src.telemetry.metrics import StepMetrics

EXPECTED_REWARD_THRESHOLDS: Final[tuple[float, ...]] = (
    -1.0,
    0.0,
    0.5,
)
CONFIDENCE_THRESHOLDS: Final[tuple[float, ...]] = (
    0.35,
    0.45,
    0.55,
)
RELIABILITY_THRESHOLDS: Final[tuple[float, ...]] = (
    0.45,
    0.55,
    0.65,
)

ARBITRATION_GATE_FIELDS: Final[tuple[str, ...]] = (
    "channel",
    "min_expected_reward",
    "min_confidence",
    "min_reliability",
    "step_count",
    "usable_disagreement_count",
    "candidate_count",
    "rescue_candidate_count",
    "override_risk_count",
    "neutral_candidate_count",
    "candidate_rate",
    "candidate_capture_rate",
    "rescue_candidate_rate",
    "override_risk_rate",
    "neutral_candidate_rate",
    "readiness_score",
    "mean_rule_reward_when_candidate",
    "mean_expected_reward_when_candidate",
    "mean_confidence_when_candidate",
    "mean_reliability_when_candidate",
)

ActionGetter = Callable[[StepMetrics], Action | None]
UsableGetter = Callable[[StepMetrics], bool]
FloatGetter = Callable[[StepMetrics], float]



def write_arbitration_gate_sweep_csv(
    history: Sequence[StepMetrics],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = summarize_arbitration_gate_sweep(history)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=ARBITRATION_GATE_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)



def summarize_arbitration_gate_sweep(
    history: Sequence[StepMetrics],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []

    rows.extend(
        summarize_channel_thresholds(
            channel="same_run_episodic",
            history=history,
            action_getter=lambda item: item.episodic_same_run_action,
            usable_getter=lambda item: item.episodic_same_run_is_usable,
            expected_getter=lambda item: item.episodic_same_run_expected_reward,
            confidence_getter=lambda item: item.episodic_same_run_confidence,
            reliability_getter=lambda item: item.episodic_same_run_reliability,
        )
    )
    rows.extend(
        summarize_channel_thresholds(
            channel="prior_episodic",
            history=history,
            action_getter=lambda item: item.episodic_prior_action,
            usable_getter=lambda item: item.episodic_prior_is_usable,
            expected_getter=lambda item: item.episodic_prior_expected_reward,
            confidence_getter=lambda item: item.episodic_prior_confidence,
            reliability_getter=lambda item: item.episodic_prior_reliability,
        )
    )
    rows.extend(
        summarize_channel_thresholds(
            channel="combined_episodic",
            history=history,
            action_getter=lambda item: item.episodic_action,
            usable_getter=lambda item: item.episodic_is_usable,
            expected_getter=lambda item: item.episodic_expected_reward,
            confidence_getter=lambda item: item.episodic_confidence,
            reliability_getter=lambda item: item.episodic_reliability,
        )
    )

    return tuple(rows)



def summarize_channel_thresholds(
    channel: str,
    history: Sequence[StepMetrics],
    action_getter: ActionGetter,
    usable_getter: UsableGetter,
    expected_getter: FloatGetter,
    confidence_getter: FloatGetter,
    reliability_getter: FloatGetter,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []

    usable_disagreements = tuple(
        item
        for item in history
        if usable_getter(item)
        and action_getter(item) is not None
        and action_getter(item) is not item.rule_action
    )

    for min_expected_reward in EXPECTED_REWARD_THRESHOLDS:
        for min_confidence in CONFIDENCE_THRESHOLDS:
            for min_reliability in RELIABILITY_THRESHOLDS:
                candidates = tuple(
                    item
                    for item in usable_disagreements
                    if expected_getter(item) >= min_expected_reward
                    and confidence_getter(item) >= min_confidence
                    and reliability_getter(item) >= min_reliability
                )

                rows.append(
                    summarize_gate_candidates(
                        channel=channel,
                        history=history,
                        usable_disagreements=usable_disagreements,
                        candidates=candidates,
                        min_expected_reward=min_expected_reward,
                        min_confidence=min_confidence,
                        min_reliability=min_reliability,
                        expected_getter=expected_getter,
                        confidence_getter=confidence_getter,
                        reliability_getter=reliability_getter,
                    )
                )

    return tuple(rows)



def summarize_gate_candidates(
    channel: str,
    history: Sequence[StepMetrics],
    usable_disagreements: Sequence[StepMetrics],
    candidates: Sequence[StepMetrics],
    min_expected_reward: float,
    min_confidence: float,
    min_reliability: float,
    expected_getter: FloatGetter,
    confidence_getter: FloatGetter,
    reliability_getter: FloatGetter,
) -> dict[str, Any]:
    rescue_candidates = tuple(
        item
        for item in candidates
        if item.reward < 0.0
    )
    override_risks = tuple(
        item
        for item in candidates
        if item.reward > 0.0
    )
    neutral_candidates = tuple(
        item
        for item in candidates
        if item.reward == 0.0
    )

    rescue_rate = rate(
        len(rescue_candidates),
        len(candidates),
    )
    override_risk_rate = rate(
        len(override_risks),
        len(candidates),
    )
    neutral_rate = rate(
        len(neutral_candidates),
        len(candidates),
    )

    return {
        "channel": channel,
        "min_expected_reward": min_expected_reward,
        "min_confidence": min_confidence,
        "min_reliability": min_reliability,
        "step_count": len(history),
        "usable_disagreement_count": len(usable_disagreements),
        "candidate_count": len(candidates),
        "rescue_candidate_count": len(rescue_candidates),
        "override_risk_count": len(override_risks),
        "neutral_candidate_count": len(neutral_candidates),
        "candidate_rate": rate(
            len(candidates),
            len(history),
        ),
        "candidate_capture_rate": rate(
            len(candidates),
            len(usable_disagreements),
        ),
        "rescue_candidate_rate": rescue_rate,
        "override_risk_rate": override_risk_rate,
        "neutral_candidate_rate": neutral_rate,
        "readiness_score": rescue_rate - override_risk_rate,
        "mean_rule_reward_when_candidate": mean(
            item.reward
            for item in candidates
        ),
        "mean_expected_reward_when_candidate": mean(
            expected_getter(item)
            for item in candidates
        ),
        "mean_confidence_when_candidate": mean(
            confidence_getter(item)
            for item in candidates
        ),
        "mean_reliability_when_candidate": mean(
            reliability_getter(item)
            for item in candidates
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
