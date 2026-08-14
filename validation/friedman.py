import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


def variance(results):
    # Calculate mean and standard deviation over repetitions.

    
    summary = (
        results
        .groupby(["dataset", "tool", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_value",
                "std": "std_value"
            }
        )
    )

    return summary



def bootstrap(results, n_bootstrap=1000, confidence=0.95, seed=42):
    
    # Calculate bootstrap confidence intervals.

    rng = np.random.default_rng(seed)

    alpha = 1 - confidence

    output = []

    grouped = results.groupby(
        ["dataset", "tool", "metric"]
    )

    for (dataset, tool, metric), group in grouped:

        values = group["value"].values

        bootstrap_means = []

        for _ in range(n_bootstrap):

            sample = rng.choice(
                values,
                size=len(values),
                replace=True
            )

            bootstrap_means.append(
                np.mean(sample)
            )

        lower = np.percentile(
            bootstrap_means,
            100 * alpha / 2
        )

        upper = np.percentile(
            bootstrap_means,
            100 * (1 - alpha / 2)
        )

        output.append(
            {
                "dataset": dataset,
                "tool": tool,
                "metric": metric,
                "mean": np.mean(values),
                "ci_lower": lower,
                "ci_upper": upper
            }
        )

    return pd.DataFrame(output)

def friedman(results):
    
    # Perform Friedman statistical significance test.

    output = []

    grouped = results.groupby(
        ["dataset", "metric"]
    )

    for (dataset, metric), group in grouped:

        pivot = (
            group
            .pivot_table(
                index="repetition",
                columns="tool",
                values="value"
            )
            .dropna(axis=1)
        )

        # Friedman requires at least 3 tools
        if pivot.shape[1] < 3:
            continue

        statistic, p_value = friedmanchisquare(
            *[
                pivot[column].values
                for column in pivot.columns
            ]
        )

        output.append(
            {
                "dataset": dataset,
                "metric": metric,
                "chi_square": statistic,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        )

    return pd.DataFrame(output)