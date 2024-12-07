# Differential Privacy (DP)
Differential Privacy (DP) is often discussed in the context of privacy guarantees, but it's not a "metric" in the traditional sense like k-anonymity or l-diversity. Instead, it’s a framework or mechanism for ensuring privacy.

## What is Differential Privacy?
Differential Privacy is a mathematical definition that provides strong privacy guarantees by ensuring that the removal or addition of a single individual's data does not significantly affect the outcome of any analysis on a dataset. This means that an observer analyzing the results cannot confidently infer whether any individual’s data was included in the dataset.

## Why It's Different:
Not a Metric: DP isn't something you "measure" after the fact like you would with DCR or NNDR. Instead, it's a property of the algorithm used to generate or analyze data.

Mechanism, Not a Metric: DP typically involves adding noise to the output of queries or computations to mask the influence of any single data point. The level of noise depends on parameters like epsilon (ε) and delta (δ).