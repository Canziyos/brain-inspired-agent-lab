from types import SimpleNamespace

from src.core.actions import Action
from src.diagnostics.arbitration_gate_sweep_csv import (
    summarize_channel_thresholds,
)



def metric(
    reward: float,
    rule_action: Action,
    advice_action: Action | None,
    expected_reward: float,
    confidence: float,
    reliability: float,
    usable: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        reward=reward,
        rule_action=rule_action,
        advice_action=advice_action,
        advice_usable=usable,
        advice_expected_reward=expected_reward,
        advice_confidence=confidence,
        advice_reliability=reliability,
    )



def test_arbitration_gate_thresholds_filter_candidates() -> None:
    rows = summarize_channel_thresholds(
        channel="test_channel",
        history=(
            metric(
                reward=-3.0,
                rule_action=Action.MOVE_EAST,
                advice_action=Action.MOVE_NORTH,
                expected_reward=0.6,
                confidence=0.6,
                reliability=0.7,
            ),
            metric(
                reward=5.0,
                rule_action=Action.MOVE_EAST,
                advice_action=Action.MOVE_NORTH,
                expected_reward=0.2,
                confidence=0.6,
                reliability=0.7,
            ),
            metric(
                reward=-3.0,
                rule_action=Action.MOVE_EAST,
                advice_action=Action.MOVE_NORTH,
                expected_reward=0.9,
                confidence=0.2,
                reliability=0.7,
            ),
            metric(
                reward=5.0,
                rule_action=Action.MOVE_EAST,
                advice_action=Action.MOVE_EAST,
                expected_reward=0.9,
                confidence=0.9,
                reliability=0.9,
            ),
        ),
        action_getter=lambda item: item.advice_action,
        usable_getter=lambda item: item.advice_usable,
        expected_getter=lambda item: item.advice_expected_reward,
        confidence_getter=lambda item: item.advice_confidence,
        reliability_getter=lambda item: item.advice_reliability,
    )

    by_threshold = {
        (
            row["min_expected_reward"],
            row["min_confidence"],
            row["min_reliability"],
        ): row
        for row in rows
    }

    permissive = by_threshold[(0.0, 0.45, 0.65)]
    assert permissive["candidate_count"] == 2
    assert permissive["rescue_candidate_count"] == 1
    assert permissive["override_risk_count"] == 1
    assert permissive["readiness_score"] == 0.0

    stricter = by_threshold[(0.5, 0.45, 0.65)]
    assert stricter["candidate_count"] == 1
    assert stricter["rescue_candidate_count"] == 1
    assert stricter["override_risk_count"] == 0
    assert stricter["readiness_score"] == 1.0
