#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs a benchmark experiment using Colosseum framework.
It specifically uses the episodic communicating benchmark.
"""

import os
from colosseum.analysis.tables import get_latex_table_of_average_indicator
from colosseum.analysis.tables import get_latex_table_of_indicators
from colosseum.benchmark.benchmark import ColosseumDefaultBenchmark
from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_benchmark
from colosseum.experiment.experiment_instances import run_experiment_instances
from colosseum.agent.agents.episodic.q_learning import QLearningEpisodic
from colosseum.agent.agents.episodic.posterior_sampling import PSRLEpisodic
from colosseum.agent.agents.episodic.dqn import DQNEpisodic
from colosseum.agent.agents.episodic.boot_dqn import BootDQNEpisodic
from colosseum.analysis.plots import (
    agent_performances_per_mdp_plot,
    plot_indicator_in_hardness_space,
    plot_indicator_in_hardness_space_3d,
)


# Create a dictionary mapping agent class to its gin config
def get_agent_configs(tabular=True):
    """
    Returns a dictionary mapping agent classes to their gin configurations
    Only using episodic agents to ensure type compatibility

    Parameters
    ----------
    tabular : bool
        If True, return only tabular agents. If False, return only non-tabular agents.
    """

    # Tabular agents (QLearning and PSRL)
    tabular_agents = {
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
        from colosseum.agent.mdp_models import bayesian_models
        prms_0/PSRLEpisodic.reward_prior_model = %bayesian_models.RewardsConjugateModel.N_NIG
        prms_0/PSRLEpisodic.transitions_prior_model = %bayesian_models.TransitionsConjugateModel.M_DIR
        prms_0/PSRLEpisodic.rewards_prior_prms = [1.0, 1, 1, 1]
        prms_0/PSRLEpisodic.transitions_prior_prms = [1.0]
        """,
    }

    # Neural network-based agents (DQN and BootDQN)
    non_tabular_agents = {
        # DQN agent
        DQNEpisodic: """
            prms_0/DQNEpisodic.network_width = 64
            prms_0/DQNEpisodic.network_depth = 2
            prms_0/DQNEpisodic.batch_size = 32
            prms_0/DQNEpisodic.sgd_period = 1
            prms_0/DQNEpisodic.target_update_period = 4
            prms_0/DQNEpisodic.epsilon = 0.05
            """,
        # Bootstrap DQN agent
        BootDQNEpisodic: """
            prms_0/BootDQNEpisodic.network_width = 64
            prms_0/BootDQNEpisodic.network_depth = 2
            prms_0/BootDQNEpisodic.batch_size = 32
            prms_0/BootDQNEpisodic.sgd_period = 1
            prms_0/BootDQNEpisodic.target_update_period = 4
            prms_0/BootDQNEpisodic.mask_prob = 0.5
            prms_0/BootDQNEpisodic.noise_scale = 0.0
            prms_0/BootDQNEpisodic.n_ensemble = 5
            """,
    }

    # Based on examining the agents' code, we can see that:
    # - QLearningEpisodic.is_emission_map_accepted() method checks emission_map.is_tabular
    # - PSRLEpisodic.is_emission_map_accepted() method also requires emission_map.is_tabular
    # Therefore, these agents can only work with tabular emission maps

    return tabular_agents if tabular else non_tabular_agents


def main():
    """
    Main function to run both tabular and non-tabular benchmarks.
    """
    results = {}

    from colosseum.config import enable_multiprocessing
    enable_multiprocessing(max_cores=16)

    # Run tabular experiment
    print("\n===== RUNNING TABULAR EXPERIMENT =====\n")
    tabular_results = run_benchmark_experiment(tabular=True)
    results["tabular"] = tabular_results

    from colosseum.config import enable_multiprocessing
    enable_multiprocessing(max_cores=4)

    # Run non-tabular experiment
    print("\n===== RUNNING NON-TABULAR EXPERIMENT =====\n")
    non_tabular_results = run_benchmark_experiment(tabular=False)
    results["non_tabular"] = non_tabular_results

    return results


def run_benchmark_experiment(tabular=True):
    """
    Run a benchmark experiment with either tabular or non-tabular agents.

    Parameters
    ----------
    tabular : bool
        If True, run experiment with tabular agents and tabular emission map.
        If False, run experiment with non-tabular agents and non-tabular emission map.

    Returns
    -------
    dict
        The results of the experiment.
    """
    # Step 1: Create the benchmark object using the predefined enum
    benchmark_postfix = "tabular" if tabular else "non_tabular"
    benchmark = ColosseumDefaultBenchmark.EPISODIC_COMMUNICATING.get_benchmark(
        postfix=benchmark_postfix, non_tabular=not tabular
    )

    print(f"Running benchmark: {benchmark.name}")
    print(f"MDPs in benchmark: {list(benchmark.mdps_gin_configs.keys())}")
    print(f"Emission map: {'Tabular' if tabular else 'Non-tabular'}")

    # Step 2: Get agent configurations based on experiment type
    agents_configs = get_agent_configs(tabular=tabular)
    if not agents_configs:
        print("No compatible agents for this experiment type.")
        return None

    print(f"Agents to evaluate: {list(agents_configs.keys())}")

    # Step 3: Instantiate the benchmark and get experiment instances

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
    experiment_folder = benchmark.get_experiments_benchmark_log_folder()
    print(f"Experiments completed. Results available in: {experiment_folder}")

    # Print simple table with regret information
    print("\nRegret Performance Table:")
    regret_table = get_latex_table_of_average_indicator(
        experiment_folder=experiment_folder,
        indicator="normalized_cumulative_regret",
        print_table=True,
    )

    # Save performance report
    save_performance_report(experiment_folder)

    print("\nDetailed performance tables have been saved to reports directory.")

    return results


def save_performance_report(experiment_folder):
    """
    Generate and save a comprehensive performance report to files

    Parameters
    ----------
    experiment_folder : str
        Path to the experiment results folder
    """
    import matplotlib.pyplot as plt

    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(experiment_folder, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print(f"Generating performance reports in: {reports_dir}")

    # Generate regret performance table
    print("- Generating regret performance table...")
    regret_table = get_latex_table_of_average_indicator(
        experiment_folder=experiment_folder, 
        indicator="normalized_cumulative_regret",
        return_table=True
    )
    if isinstance(regret_table, tuple):
        regret_latex, regret_df = regret_table
    else:
        regret_latex = regret_table.to_latex(escape=False)

    # Generate reward performance table
    print("- Generating reward performance table...")
    reward_table = get_latex_table_of_average_indicator(
        experiment_folder=experiment_folder, 
        indicator="cumulative_reward",
        return_table=True
    )
    if isinstance(reward_table, tuple):
        reward_latex, reward_df = reward_table
    else:
        reward_latex = reward_table.to_latex(escape=False)

    # Generate efficiency performance table
    print("- Generating computational efficiency table...")
    efficiency_table = get_latex_table_of_average_indicator(
        experiment_folder=experiment_folder, 
        indicator="steps_per_second",
        return_table=True
    )
    if isinstance(efficiency_table, tuple):
        efficiency_latex, efficiency_df = efficiency_table
    else:
        efficiency_latex = efficiency_table.to_latex(escape=False)

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
        f.write(regret_latex)

    with open(os.path.join(reports_dir, "reward_table.tex"), "w") as f:
        f.write(reward_latex)

    with open(os.path.join(reports_dir, "efficiency_table.tex"), "w") as f:
        f.write(efficiency_latex)

    with open(os.path.join(reports_dir, "detailed_table.tex"), "w") as f:
        f.write(detailed_table)

    # Generate and save performance plots
    print("- Generating agent performance plots...")

    # Plot regret performance across MDPs
    regret_fig = agent_performances_per_mdp_plot(
        experiment_folder=experiment_folder,
        indicator="normalized_cumulative_regret",
        figsize_scale=6,
        standard_error=True,
        savefig_folder=reports_dir,
    )

    # Plot reward performance across MDPs
    reward_fig = agent_performances_per_mdp_plot(
        experiment_folder=experiment_folder,
        indicator="cumulative_reward",
        figsize_scale=6,
        standard_error=True,
        savefig_folder=reports_dir,
    )

    # Plot performance in hardness space
    try:
        hardness_fig = plot_indicator_in_hardness_space(
            experiment_folder=experiment_folder,
            indicator="normalized_cumulative_regret",
            fig_size=8,
            savefig_folder=reports_dir,
        )
        hardness_fig = plot_indicator_in_hardness_space_3d(
            experiment_folder=experiment_folder,
            indicator="normalized_cumulative_regret",
            fig_size=8,
            savefig_folder=reports_dir,
        )
    except Exception as e:
        print(f"Note: Could not generate hardness space plot. Reason: {e}")

    print("All performance reports and plots successfully saved!")


if __name__ == "__main__":
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # This ensures the script runs only when executed directly
    results = main()