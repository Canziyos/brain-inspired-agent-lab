from types import SimpleNamespace

from src.core.actions import Action
from src.diagnostics.advisor_evaluation_csv import (
    summarize_advisor_evaluation,
)



def metric(
    reward: float,
    rule_action: Action,
    combined_action: Action | None,
    same_run_action: Action | None,
    prior_action: Action | None,
    imagination_action: Action,
    combined_usable: bool = False,
    same_run_usable: bool = False,
    prior_usable: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        reward=reward,
        rule_action=rule_action,
        episodic_action=combined_action,
        episodic_is_usable=combined_usable,
        episodic_same_run_action=same_run_action,
        episodic_same_run_expected_reward=1.5,
        episodic_same_run_confidence=0.4,
        episodic_same_run_reliability=0.5,
        episodic_same_run_is_usable=same_run_usable,
        episodic_prior_action=prior_action,
        episodic_prior_expected_reward=1.25,
        episodic_prior_confidence=0.45,
        episodic_prior_reliability=0.55,
        episodic_prior_is_usable=prior_usable,
        imagination_action=imagination_action,
        episodic_expected_reward=2.0,
        episodic_confidence=0.5,
        episodic_reliability=0.6,
        imagination_expected_reward=3.0,
    )



def test_summarize_advisor_evaluation_compares_channels() -> None:
    rows = summarize_advisor_evaluation(
        (
            metric(
                reward=5.0,
                rule_action=Action.MOVE_EAST,
                combined_action=Action.MOVE_EAST,
                same_run_action=Action.MOVE_EAST,
                prior_action=Action.MOVE_NORTH,
                imagination_action=Action.MOVE_EAST,
                combined_usable=True,
                same_run_usable=True,
                prior_usable=True,
            ),
            metric(
                reward=-1.0,
                rule_action=Action.MOVE_NORTH,
                combined_action=Action.MOVE_EAST,
                same_run_action=None,
                prior_action=Action.MOVE_NORTH,
                imagination_action=Action.MOVE_EAST,
                combined_usable=True,
                prior_usable=True,
            ),
        )
    )

    by_channel = {
        row["channel"]: row
        for row in rows
    }

    assert set(by_channel) == {
        "rule_policy",
        "same_run_episodic",
        "prior_episodic",
        "combined_episodic",
        "neural_imagination",
    }

    assert by_channel["rule_policy"]["agreement_rate"] == 1.0
    assert by_channel["combined_episodic"]["advice_count"] == 2
    assert by_channel["combined_episodic"]["agreement_count"] == 1
    assert by_channel["combined_episodic"]["agreement_rate"] == 0.5
    assert by_channel["same_run_episodic"]["advice_count"] == 1
    assert by_channel["same_run_episodic"]["mean_expected_reward"] == 1.5
    assert by_channel["same_run_episodic"]["mean_confidence"] == 0.4
    assert by_channel["same_run_episodic"]["mean_reliability"] == 0.5
    assert by_channel["prior_episodic"]["usable_agreement_rate"] == 0.5
    assert by_channel["prior_episodic"]["mean_expected_reward"] == 1.25
    assert by_channel["prior_episodic"]["mean_confidence"] == 0.45
    assert by_channel["prior_episodic"]["mean_reliability"] == 0.55
    assert by_channel["combined_episodic"][
        "mean_rule_reward_when_advice_agreed"
    ] == 5.0
    assert by_channel["combined_episodic"][
        "mean_rule_reward_when_advice_disagreed"
    ] == -1.0
