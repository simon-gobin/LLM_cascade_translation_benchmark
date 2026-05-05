# LLM_cascade_translation_benchmark

Benchmark and notebook workspace for an Irish-to-English cascaded speech translation assignment.

## Colab Setup

Clone this repository and the dataset repository separately in Colab:

```bash
git clone https://github.com/simon-gobin/LLM_cascade_translation_benchmark.git
cd LLM_cascade_translation_benchmark
git clone https://github.com/shashwatup9k/iwslt2023_ga-eng.git
```

The dataset is intentionally not stored in this repository because it is large.

## Main Files

- `Student_Colab_Notebook_in_progress.ipynb`: working notebook
- `pipeline_utils.py`: reusable functions extracted from the notebook
- `benchmark.py`: experiment runner for batch benchmarking
- `benchmark_sources.md`: source links for implementation / report writing
