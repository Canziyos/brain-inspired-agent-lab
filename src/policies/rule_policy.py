import random
from collections.abc import Sequence

from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.core.motivation import (
    MEDIUM_ENERGY,
    food_action_reason,
    food_motivation,
    mystery_motivation,
    rest_motivation,
)
from src.core.perception import Observation
from src.core.world import CellType
from src.planning.goal_planner import (
    GoalKind,
    GoalPlan,
)


EMPTY_REVISIT_SCORE = 5.0
EMPTY_EXPLORE_SCORE = 15.0

FRONTIER_TRAVEL_SCORE = 14.0
MYSTERY_TRAVEL_SCORE = 18.0
REST_ALLOWED_BELOW_ENERGY = MEDIUM_ENERGY


def planned_goal_score(
    agent: Agent,
    plan: GoalPlan,
) -> float:
    if plan.score is not None:
        return plan.score

    state = agent.snapshot()

    if plan.kind is GoalKind.FOOD:
        return food_motivation(state)

    if plan.kind is GoalKind.MYSTERY:
        return max(
            MYSTERY_TRAVEL_SCORE,
            mystery_motivation(state),
        )

    if plan.kind is GoalKind.FRONTIER:
        return FRONTIER_TRAVEL_SCORE

    raise ValueError(f"Unsupported goal kind: {plan.kind!r}")


def evaluate_observation(
    agent: Agent,
    observation: Observation,
) -> ActionEvaluation | None:
    state = agent.snapshot()
    target_position = observation.x, observation.y
    visited = target_position in agent.visited

    action = Action.from_positions(
        agent.position,
        target_position,
    )

    if observation.cell is CellType.FOOD:
        return ActionEvaluation(
            action=action,
            policy_score=food_motivation(state),
            rationale=food_action_reason(state),
        )

    if observation.cell is CellType.DANGER:
        return None

    if observation.cell is CellType.MYSTERY:
        return ActionEvaluation(
            action=action,
            policy_score=mystery_motivation(state),
            rationale="investigate mystery",
        )

    if observation.cell is CellType.EMPTY:
        return ActionEvaluation(
            action=action,
            policy_score=(
                EMPTY_REVISIT_SCORE
                if visited
                else EMPTY_EXPLORE_SCORE
            ),
            rationale=(
                "revisit empty cell"
                if visited
                else "explore empty cell"
            ),
        )

    return None


def rest_evaluation(agent: Agent) -> ActionEvaluation:
    return ActionEvaluation(
        action=Action.REST,
        policy_score=rest_motivation(agent.snapshot()),
        rationale="recover energy",
    )


def should_consider_rest(
    agent: Agent,
    movement_evaluations: Sequence[ActionEvaluation],
) -> bool:
    if not movement_evaluations:
        return True

    return agent.energy < REST_ALLOWED_BELOW_ENERGY


def planned_action_matches(
    agent: Agent,
    action: Action,
    plan: GoalPlan,
) -> bool:
    return (
        action.is_move
        and action.target_from(agent.position) == plan.next_step
    )


def apply_goal_plan(
    agent: Agent,
    evaluations: Sequence[ActionEvaluation],
    plan: GoalPlan | None,
) -> tuple[ActionEvaluation, ...]:
    if plan is None:
        return tuple(evaluations)

    plan_score = planned_goal_score(
        agent=agent,
        plan=plan,
    )

    updated: list[ActionEvaluation] = []

    for evaluation in evaluations:
        if planned_action_matches(
            agent=agent,
            action=evaluation.action,
            plan=plan,
        ):
            updated.append(
                ActionEvaluation(
                    action=evaluation.action,
                    policy_score=max(
                        evaluation.policy_score,
                        plan_score,
                    ),
                    rationale=(
                        f"follow {plan.kind.value} goal "
                        f"toward {plan.target}"
                    ),
                )
            )
        else:
            updated.append(evaluation)

    return tuple(updated)


def evaluate_actions(
    agent: Agent,
    observations: Sequence[Observation],
    plan: GoalPlan | None = None,
) -> tuple[ActionEvaluation, ...]:
    movement_evaluations: list[ActionEvaluation] = []

    for observation in observations:
        evaluation = evaluate_observation(
            agent=agent,
            observation=observation,
        )

        if evaluation is not None:
            movement_evaluations.append(evaluation)

    evaluations: list[ActionEvaluation] = []

    if should_consider_rest(
        agent=agent,
        movement_evaluations=movement_evaluations,
    ):
        evaluations.append(rest_evaluation(agent))

    evaluations.extend(movement_evaluations)

    return apply_goal_plan(
        agent=agent,
        evaluations=evaluations,
        plan=plan,
    )


def choose_action(
    evaluations: Sequence[ActionEvaluation],
    rng: random.Random | None = None,
) -> ActionEvaluation:
    if not evaluations:
        raise ValueError("Baby Vice has no actions.")

    best_score = max(
        evaluation.policy_score
        for evaluation in evaluations
    )

    best_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation.policy_score == best_score
    ]

    if rng is None:
        return random.choice(best_evaluations)

    return rng.choice(best_evaluations)
