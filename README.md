# TDSbenchmark
This repository benchmarks tools for tabular data synthesis, providing insights and comparisons to help users identify the most suitable tool for generating synthetic data tailored to their specific use case.
This repository is the code for the paper: **Benchmarking Tabular Data Synthesis: Evaluating Tools, Metrics, and Datasets on Commodity Hardware for End-Users**.

## Necessary files per tool
Each tool needs a repo with _toolname_-main and inside:
- model code (added here using git submodule)
- run_tool.py
- python-version.txt
- requirements.txt
- special-torch.txt (optional for tools that need special torch library before installing reqs.)

Put the real dataset inside the data folder as a file with .csv extension.
If you want to compare several tools, pre-processing the original dataset and scaling it is advised.

## How to run
Inside TDSbenchmark run:
```
python benchmark.py
```
Then, as prompted, provide the experiment .json file. For examples of .json configuration files see the "experiments" folder.

The benchmark will run te shell script "monitor_usage" to create a .csv file with the resource performance (CPU, GPU, memory, time) during the benchmark. For iOS, use monitor_usageMac.sh.


### Example
```
python bechmark.py
experiments/per_dataset/adult.json
```

### Results
When finished, benchmark saves:
- A fake dataset under fake_datasets/_toolname_/_toolname_ _dataset_.csv
- Performance files: one for CPU and memory performance, one for GPU performance and several for other evaluation metrics. For a complete list of the evaluation metrics, see the original paper.
- For result plots of our benchmark, see "results".



