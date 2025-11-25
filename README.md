# Harmful-RAG-based-bots
Harmful RAG Based Bots: Uncovering The Impact of Public Sources

This project investigates how RAG-based AI assistants can be manipulated through adversarial content injected into public, freely commentable sources. We simulate realistic adversarial user comments using automated methods such as AutoDAN to analyze how poisoned content influences retrieval pipelines and leads to potentially unsafe or harmful LLM responses.

Our experiments span multiple retrievers and large language models under varying poisoning budgets. By evaluating how different systems respond to targeted injections, we aim to reveal the mechanisms that make RAG systems vulnerable and quantify the impact on both retrieval quality and downstream generation.

The findings emphasize the need for strong content-filtering pipelines and poisoning-aware retrieval strategies. Ultimately, this work contributes to building safer, more reliable, and more trustworthy RAG-based AI assistants.

## Usage

Run the RAG chatbot from the `chatbot` folder. The script now accepts a `--model` argument (required) and an optional `--query`:

```bash
python3 chatbot/main.py --model llama4
python3 chatbot/main.py --model llama4 --query "Will this game run on Windows 7?"
```

The `--model` value is a short key that the script maps to environment variables. For example, using `--model llama4` the script will look for `LLAMA4_MODEL_NAME` (and optionally `LLAMA4_MODEL_LOCATION`).

## .env and configuration

Create a `.env` file at the repository root (not committed) and fill in the values. Use `.env.example` as a template. Important variables:

- `PROJECT_ID` — your GCP project id
- `MODEL_LOCATION` — default model location (eg. `us-central1`) used if a per-model location is not present
- `REGION` — Discovery Engine region for your data store (eg. `us`)
- `DATA_STORE_ID` — Discovery Engine data store id
- `<PREFIX>_MODEL_NAME` — MaaS model name for a given model key, where `<PREFIX>` is the uppercase model key (e.g. `NAME`)
- `<PREFIX>_MODEL_LOCATION` — optional per-model location that overrides `MODEL_LOCATION` (e.g. `LLAMA4_MODEL_LOCATION`)

See `.env.example` for a full example that includes placeholder entries for `llama4`, `gemini_2.5_flash`

## Notes
- Keep the real `.env` out of version control. `.env.example` is provided as a template only.
- Make sure you have Application Default Credentials set up (`gcloud auth application-default login`) and that Vertex AI and Discovery Engine APIs are enabled in your GCP project.
