from dataclasses import dataclass

from src.configs import SimulationConfig
from src.simulation.session import SimulationSession
from src.simulation.step import run_simulation_step
from src.telemetry.metrics import StepMetrics


@dataclass(frozen=True, slots=True)
class SimulationRunResult:
    history: list[StepMetrics]
    terminated: bool
    truncated: bool



def run_simulation_loop(
    config: SimulationConfig,
    session: SimulationSession,
) -> SimulationRunResult:
    history: list[StepMetrics] = []

    terminated = False
    truncated = False

    while not terminated and not truncated:
        (
            metrics,
            terminated,
            truncated,
            session.outcome_neural_state,
        ) = run_simulation_step(
            step=len(history),
            config=config,
            env=session.env,

            reward_network=session.reward_network,
            reward_optimizer=session.reward_optimizer,
            reward_samples=session.reward_samples,

            outcome_model=session.outcome_model,
            outcome_optimizer=session.outcome_optimizer,
            outcome_samples=session.outcome_samples,

            working_memory=session.working_memory,
            episodic_trace=session.episodic_trace,
            prior_episodes=session.prior_episodes,

            policy_rng=session.policy_rng,
            training_rng=session.reward_training_rng,
            outcome_training_rng=session.outcome_training_rng,

            outcome_neural_state=session.outcome_neural_state,
        )

        history.append(metrics)

    return SimulationRunResult(
        history=history,
        terminated=terminated,
        truncated=truncated,
    )
