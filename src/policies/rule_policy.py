import random

from src.core.actions import Action, ActionEvaluation
from src.core.agent import Agent
from src.core.perception import Observation
from src.core.world import DANGER, EMPTY, FOOD, MYSTERY
from src.planning.goal_planner import (
    GoalKind,
    GoalPlan,
)

FRONTIER_TRAVEL_SCORE = 14.0
MYSTERY_TRAVEL_SCORE = 18.0


def planned_goal_score(
    agent: Agent,
    kind: GoalKind,
) -> float:
    if kind is GoalKind.FOOD:
        if agent.energy < 30.0:
            return 90.0

        if agent.energy < 60.0:
            return 70.0

        if agent.energy < 85.0:
            return 40.0

        return 16.0

    if kind is GoalKind.MYSTERY:
        return max(
            MYSTERY_TRAVEL_SCORE,
            agent.curiosity * 0.6,
        )

    return FRONTIER_TRAVEL_SCORE


def evaluate_actions(
    agent: Agent,
    observations: list[Observation],
    plan: GoalPlan | None = None,
) -> list[ActionEvaluation]:
    evaluations: list[ActionEvaluation] = []

    if agent.energy < 30:
        rest_score = 60.0
    elif agent.energy < 60:
        rest_score = 25.0
    elif agent.energy < 85:
        rest_score = 8.0
    else:
        rest_score = 0.0

    evaluations.append(
        ActionEvaluation(
            action=Action(
                kind="rest",
                target_x=agent.x,
                target_y=agent.y,
            ),
            score=rest_score,
            reason="recover energy",
        )
    )

    for observation in observations:
        x = observation.x
        y = observation.y
        cell = observation.cell
        visited = (x, y) in agent.visited

        action = Action(
            kind="move",
            target_x=x,
            target_y=y,
        )

        if cell == FOOD:
            if agent.energy < 30:
                score = 90.0
                reason = "desperate for food"
            elif agent.energy < 60:
                score = 70.0
                reason = "hungry, food useful"
            elif agent.energy < 85:
                score = 40.0
                reason = "food nearby"
            else:
                score = 12.0
                reason = "food nearby, but already full"

        elif cell == DANGER:
            continue

        elif cell == MYSTERY:
            score = agent.curiosity * 0.6
            reason = "investigate mystery"

        elif cell == EMPTY:
            score = 5.0 if visited else 15.0
            reason = (
                "revisit empty cell"
                if visited
                else "explore empty cell"
            )

        else:
            continue

        evaluations.append(
            ActionEvaluation(
                action=action,
                score=score,
                reason=reason,
            )
        )

    if plan is not None:
        planned_step = plan.next_step
        plan_score = planned_goal_score(
            agent,
            plan.kind,
        )

        for index, evaluation in enumerate(
            evaluations
        ):
            action = evaluation.action

            matches_plan = (
                action.kind == "move"
                and (
                    action.target_x,
                    action.target_y,
                ) == planned_step
            )

            if not matches_plan:
                continue

            evaluations[index] = ActionEvaluation(
                action=action,
                score=max(
                    evaluation.score,
                    plan_score,
                ),
                reason=(
                    f"follow {plan.kind.value} goal "
                    f"toward {plan.target}"
                ),
            )

            break

    return evaluations


def choose_action(
    evaluations: list[ActionEvaluation],
    rng: random.Random | None = None,
) -> ActionEvaluation:
    if not evaluations:
        raise ValueError("Baby Vice has no actions.")

    best_score = max(
        evaluation.score
        for evaluation in evaluations
    )

    best_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation.score == best_score
    ]

    random_source = (
        rng
        if rng is not None
        else random
    )

    return random_source.choice(
        best_evaluations
    )