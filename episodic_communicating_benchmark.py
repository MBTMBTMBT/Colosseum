#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs a benchmark experiment using Colosseum framework.
It specifically uses the episodic communicating benchmark.
"""

import os
from colosseum.benchmark.benchmark import ColosseumDefaultBenchmark
from colosseum.experiment.experiment_instances import run_experiment_instances
from colosseum.agent.agents.episodic.q_learning import QLearningEpisodic
from colosseum.agent.agents.episodic.posterior_sampling import PSRLEpisodic
from colosseum.agent.agents.episodic.dqn import DQNEpisodic
from colosseum.agent.agents.episodic.boot_dqn import BootDQNEpisodic
from colosseum.agent.agents.episodic.actor_critic import ActorCriticEpisodic
from colosseum.agent.agents.random import RandomAgentEpisodic


# Create a dictionary mapping agent class to its gin config
def get_agent_configs():
    """
    Returns a dictionary mapping agent classes to their gin configurations
    Only using episodic agents to ensure type compatibility
    """

    # Create a dictionary with multiple agents and their configurations
    agents_configs = {
        # Q-Learning agent
        QLearningEpisodic: """
        prms_0/QLearningEpisodic.p = 0.05
        prms_0/QLearningEpisodic.c_1 = 0.5
        prms_0/QLearningEpisodic.c_2 = 0.5
        prms_0/QLearningEpisodic.min_at = 0.1
        prms_0/QLearningEpisodic.UCB_type = "bernstein"
        """,
        # PSRL agent
        PSRLEpisodic: """
        prms_0/PSRLEpisodic.alpha = 1.0
        prms_0/PSRLEpisodic.beta = 1.0
        """,
        # DQN agent
        DQNEpisodic: """
        prms_0/DQNEpisodic.learning_rate = 0.001
        prms_0/DQNEpisodic.batch_size = 32
        prms_0/DQNEpisodic.hidden_size = 64
        prms_0/DQNEpisodic.buffer_size = 10000
        prms_0/DQNEpisodic.target_update_freq = 100
        """,
        # Bootstrap DQN agent
        BootDQNEpisodic: """
        prms_0/BootDQNEpisodic.network_width = 64
        prms_0/BootDQNEpisodic.network_depth = 2
        prms_0/BootDQNEpisodic.batch_size = 32
        prms_0/BootDQNEpisodic.sgd_period = 1
        prms_0/BootDQNEpisodic.target_update_period = 4
        prms_0/BootDQNEpisodic.mask_prob = 0.8
        prms_0/BootDQNEpisodic.noise_scale = 0.0
        prms_0/BootDQNEpisodic.n_ensemble = 8
        """,
        # Actor-Critic agent
        ActorCriticEpisodic: """
        prms_0/ActorCriticEpisodic.batch_size = 32
        prms_0/ActorCriticEpisodic.discount = 0.99
        prms_0/ActorCriticEpisodic.learning_rate = 0.001
        prms_0/ActorCriticEpisodic.entropy_cost = 0.01
        prms_0/ActorCriticEpisodic.baseline_cost = 0.5
        """,
        # Random agent (episodic version)
        RandomAgentEpisodic: """
        prms_0/RandomAgentEpisodic.seed = 42
        """,
    }

    return agents_configs


def main():
    """
    Main function to run the benchmark.
    """
    # Step 1: Create the benchmark object using the predefined enum
    # Use EPISODIC_COMMUNICATING which corresponds to value 3 in the enum
    benchmark = ColosseumDefaultBenchmark.EPISODIC_COMMUNICATING.get_benchmark()

    print(f"Running benchmark: {benchmark.name}")
    print(f"MDPs in benchmark: {list(benchmark.mdps_gin_configs.keys())}")

    # Step 2: Get agent configurations
    agents_configs = get_agent_configs()
    print(f"Agents to evaluate: {list(agents_configs.keys())}")

    # Step 3: Instantiate the benchmark and get experiment instances
    # Import after definitions to avoid circular imports
    from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_benchmark

    # Set overwrite_previous_experiment=True to force recreation of experiment folder
    experiment_instances = instantiate_and_get_exp_instances_from_benchmark(
        agents_configs=agents_configs,
        benchmark=benchmark,
        overwrite_previous_experiment=True,
    )

    print(f"Created {len(experiment_instances)} experiment instances")

    # Step 4: Run the experiments
    # This will execute all experiments and collect results
    results = run_experiment_instances(experiment_instances)

    # Step 5: Generate and print summary tables
    print(
        f"Experiments completed. Results available in: {benchmark.get_experiments_benchmark_log_folder()}"
    )

    # Import analysis tools
    from colosseum.analysis.tables import get_latex_table_of_average_indicator
    from colosseum.analysis.tables import get_latex_table_of_indicators

    # Generate performance tables for standard metrics
    experiment_folder = benchmark.get_experiments_benchmark_log_folder()

    # Print simple table with regret information
    print("\nRegret Performance Table:")
    regret_table = get_latex_table_of_average_indicator(
        experiment_folder=experiment_folder,
        indicator="normalized_cumulative_regret",
        print_table=True,
    )

    # Print multi-indicator table with more detailed information
    print("\nDetailed Performance Table:")
    indicators_table = get_latex_table_of_indicators(
        experiment_folder=experiment_folder,
        indicators=[
            "normalized_cumulative_regret",
            "cumulative_reward",
            "steps_per_second",
        ],
        print_table=True,
    )

    return results


def save_performance_report(experiment_folder):
    """
    Generate and save a comprehensive performance report to files

    Parameters
    ----------
    experiment_folder : str
        Path to the experiment results folder
    """
    import os
    from colosseum.analysis.tables import get_latex_table_of_average_indicator
    from colosseum.analysis.tables import get_latex_table_of_indicators

    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(experiment_folder, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print(f"Generating performance reports in: {reports_dir}")

    # Generate regret performance table
    print("- Generating regret performance table...")
    regret_table = get_latex_table_of_average_indicator(
        experiment_folder=experiment_folder, indicator="normalized_cumulative_regret"
    )

    # Generate reward performance table
    print("- Generating reward performance table...")
    reward_table = get_latex_table_of_average_indicator(
        experiment_folder=experiment_folder, indicator="cumulative_reward"
    )

    # Generate efficiency performance table
    print("- Generating computational efficiency table...")
    efficiency_table = get_latex_table_of_average_indicator(
        experiment_folder=experiment_folder, indicator="steps_per_second"
    )

    # Generate detailed multi-indicator table
    print("- Generating detailed multi-indicator table...")
    detailed_table = get_latex_table_of_indicators(
        experiment_folder=experiment_folder,
        indicators=[
            "normalized_cumulative_regret",
            "cumulative_reward",
            "steps_per_second",
        ],
    )

    # Save tables to files
    with open(os.path.join(reports_dir, "regret_table.tex"), "w") as f:
        f.write(regret_table)

    with open(os.path.join(reports_dir, "reward_table.tex"), "w") as f:
        f.write(reward_table)

    with open(os.path.join(reports_dir, "efficiency_table.tex"), "w") as f:
        f.write(efficiency_table)

    with open(os.path.join(reports_dir, "detailed_table.tex"), "w") as f:
        f.write(detailed_table)

    print("All performance reports successfully saved!")


if __name__ == "__main__":
    # This ensures the script runs only when executed directly
    results = main()

    # Generate and save performance reports
    benchmark = ColosseumDefaultBenchmark.EPISODIC_COMMUNICATING.get_benchmark()
    save_performance_report(benchmark.get_experiments_benchmark_log_folder())
