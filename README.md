# Harmful-RAG-based-bots
Harmful RAG Based Bots: Uncovering The Impact of Public Sources

This project investigates how RAG-based AI assistants can be manipulated through adversarial content injected into public, freely commentable sources. We simulate realistic adversarial user comments using automated methods such as AutoDAN to analyze how poisoned content influences retrieval pipelines and leads to potentially unsafe or harmful LLM responses.

Our experiments span multiple retrievers and large language models under varying poisoning budgets. By evaluating how different systems respond to targeted injections, we aim to reveal the mechanisms that make RAG systems vulnerable and quantify the impact on both retrieval quality and downstream generation.

The findings emphasize the need for strong content-filtering pipelines and poisoning-aware retrieval strategies. Ultimately, this work contributes to building safer, more reliable, and more trustworthy RAG-based AI assistants.

## Usage

1. Clearing the dataset
Makes the dataset compatible with our usecase.
```bash
uv run dataset/clear_data.py
```

2. Updating the data-store and rebuilding the RAG
This script automates the full update cycle for your Vertex AI Search Data Store: Upload $\rightarrow$ Purge $\rightarrow$ Import.
```bash
uv run rag/rebuild_rag.py --source-file dataset/filtered_QA_Video_Games.jsonl
```

3. Interacting with the RAG and Model
Run the RAG chatbot from the `chatbot` folder. The script now accepts a `--model` argument (required) and an optional `--query`:

```bash
uv run chatbot/main.py --log-level debug --model llama4 --prompt-format "query_before_context" --no-system-prompt --output-file output/test.jsonl --num-queries 10 --prompt-dataset-file dataset/filtered_QA_with_injections_10pct.jsonl
```

The `--model` value is a short key that the script maps to environment variables. For example, using `--model llama4` the script will look for environment variables `LLAMA4_MODEL_NAME` and `LLAMA4_MODEL_LOCATION`.

## .env and configuration

Create a `.env` file at the repository root (not committed) and fill in the values. Use `.env.example` as a template. Important variables:

- `PROJECT_ID` — your GCP project id
- `MODEL_LOCATION` — default model location (eg. `us-central1`) used if a per-model location is not present
- `SEARCH_REGION` — Discovery Engine region for your data store (eg. `us`)
- `DATA_STORE_ID` — Discovery Engine data store id
- `<PREFIX>_MODEL_NAME` — MaaS model name for a given model key, where `<PREFIX>` is the uppercase model key (e.g. `NAME`)
- `<PREFIX>_MODEL_LOCATION` — optional per-model location that overrides `MODEL_LOCATION` (e.g. `LLAMA4_MODEL_LOCATION`)

See `.env.example` for a full example that includes placeholder entries for `llama4`, `gemini_2.5_flash`

## Notes
- Keep the real `.env` out of version control. `.env.example` is provided as a template only.
- Make sure you have Application Default Credentials set up (`gcloud auth application-default login`) and that Vertex AI and Discovery Engine APIs are enabled in your GCP project.

**Experiments**

This repository includes a set of reproducible experiments (1..7) that exercise different target / verifier model pairings and prompt configurations. Experiment orchestration is provided by `run_experiments.sh`, which runs the RAG+generation pipeline and writes a JSONL output file for each experiment: `output/experiment_<N>.jsonl` (each file contains `NUM_QUERIES`, by default 500 lines).

- **Dataset:** `dataset/filtered_QA_with_injections_10pct.jsonl` (used by `run_experiments.sh`) — this dataset contains adversarially injected prompts in a subset (10%) of entries. See `prompt_injection_scripts/` for the injection tooling.
- **Output:** `output/experiment_<N>.jsonl` — each line is a JSON object with fields like `question_id`, `model_response`, and `verification_category` (one of `benign`, `adversarial`, `refusal`).

The experiments configured in `run_experiments.sh` are as follows:

- **Experiment 1 — Normal setup (baseline)**
	- **Target model:** `llama4`
	- **Verifier model:** `gemini2.5-flash`
	- **Prompt format:** `query_before_context`
	- **System prompt enabled:** No
	- **Num queries:** 500
	- **Output path:** `output/experiment_1.jsonl`

- **Experiment 2 — System prompt importance**
	- **Target model:** `llama4`
	- **Verifier model:** `gemini2.5-flash`
	- **Prompt format:** `query_before_context`
	- **System prompt enabled:** Yes
	- **Num queries:** 500
	- **Output path:** `output/experiment_2.jsonl`

- **Experiment 3 — Ordering: context before query**
	- **Target model:** `llama4`
	- **Verifier model:** `gemini2.5-flash`
	- **Prompt format:** `context_before_query`
	- **System prompt enabled:** No
	- **Num queries:** 500
	- **Output path:** `output/experiment_3.jsonl`

- **Experiment 4 — Target model: `gemini2.5-flash` (compare targets)**
	- **Target model:** `gemini2.5-flash`
	- **Verifier model:** `gemini2.5-flash`
	- **Prompt format:** `query_before_context`
	- **System prompt enabled:** No
	- **Num queries:** 500
	- **Output path:** `output/experiment_4.jsonl`

- **Experiment 5 — Target model: `llama3.3` (compare targets)**
	- **Target model:** `llama3.3`
	- **Verifier model:** `gemini2.5-flash`
	- **Prompt format:** `query_before_context`
	- **System prompt enabled:** No
	- **Num queries:** 500
	- **Output path:** `output/experiment_5.jsonl`

- **Experiment 6 — Verifier: `llama4` (compare verifiers)**
	- **Target model:** `llama4`
	- **Verifier model:** `llama4`
	- **Prompt format:** `query_before_context`
	- **System prompt enabled:** No
	- **Num queries:** 500
	- **Output path:** `output/experiment_6.jsonl`

- **Experiment 7 — Verifier: `llama3.3` (compare verifiers)**
	- **Target model:** `llama4`
	- **Verifier model:** `llama3.3`
	- **Prompt format:** `query_before_context`
	- **System prompt enabled:** No
	- **Num queries:** 500
	- **Output path:** `output/experiment_7.jsonl`

How to run the experiments

 - Run a single experiment (example: experiment 1):
```bash
./run_experiments.sh 1
```

 - Run all experiments 1..7 sequentially:
```bash
./run_experiments.sh all
```

Notes

- If you change the dataset or injection fraction, update `DATASET` in `run_experiments.sh` accordingly.
- The experiments expect per-model environment variables (e.g., `LLAMA4_MODEL_NAME`, `LLAMA4_MODEL_LOCATION`, `GEMINI2_5_FLASH_MODEL_NAME`, etc.) to be present in your `.env` or environment. See the `.env.example` file for variable names and examples.
- Each experiment's JSONL output includes a `verification_category` field produced by the verifier model; use `scripts/plot_experiments.py` to generate summary plots of `benign` / `adversarial` / `refusal` counts.

If you want, I can also add a short summary table to the top of this README with quick command snippets for running a single experiment and plotting the results.
