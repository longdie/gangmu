# Gangmu

Gangmu is a two-stage framework for cross-document mathematical knowledge consolidation built on the Quotient-Based Graph Structure. This repository also runs the MathER-Alg evaluation, four ablation studies, and the fixed-sample Evidence Graph correctness evaluation.

## Setup

Use Python 3.11 in a Linux environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository provides `config.example.toml`. Copy it to `config.toml`, then fill in your API keys and endpoints:

```bash
cp config.example.toml config.toml
```

`config.toml` is your private local configuration and is not included in the repository.

Configure the required connection roles in `config.toml`. Each role has an API key, base URL, and protocol (`openai_chat_completions` or `claude_messages`). Environment variables such as `GANGMU_LLM_API_KEY` and `GANGMU_LLM_BASE_URL` override the corresponding values. Embedding and reranking use OpenAI-compatible embedding and `/rerank` endpoints. OCR uses a chat-completions endpoint supporting image inputs. Claude Code Agent requires a Messages-compatible endpoint and a separately installed `claude` executable (the paper used Claude Code v2.1.220).

| Role | Used for | Default model |
| --- | --- | --- |
| `llm` | PDF extraction, normalization, canonical graph construction | GPT-5.6-terra |
| `llm` | Gangmu, BM25 + SEV, four ablations | GPT-5.6-terra |
| `embedding` | Vector generation | Qwen3-Embedding-8B, 2048 dimensions |
| `reranker` | Candidate reranking | Qwen3-Reranker-8B |
| `ocr` | PDF parsing | GLM-5V-Turbo |
| `judge` | Evidence Graph correctness | Claude Opus 5 |
| `agent` | Claude Code Agent baseline | GPT-5.6-terra |

Model IDs are editable under `[models.<task>]` to match your service. Only services needed by a command are contacted. No credentials or endpoint addresses are supplied by this repository. Validate your configuration without displaying secrets with `python -m src.common.config`.

## Local data and directories

Extract the supplementary archive so that its top-level `Material/` directory is directly inside this repository:

```text
gangmu/
  Material/           # downloaded supplementary material
    MathER-Alg/
    Experiments/
  inputs/             # your PDFs
  outputs/            # constructed graphs, vector caches, runtime checkpoints
  experiments/        # newly generated evaluation results
  tmp/                # temporary tests and evaluation checkpoints
  prompts/            # all prompt templates
  scripts/            # pipeline and experiment implementations
  src/                # shared graph and model components
```

`Material/`, `inputs/`, `outputs/`, `experiments/`, `tmp/`, local credentials, and Python caches are ignored by Git. Supplementary data are read without modification. Relative command-line paths are resolved from the repository root, even when the command is invoked from another working directory.

## Run the four entrypoints

```bash
# Complete PDF-to-graph pipeline; includes PDFs in nested folders.
python main.py --input-dir inputs --output-dir outputs/documents

# Gangmu and all six reported baselines on the 512 released queries.
python mather_evaluation.py

# Only the four ablations; no separate Gangmu control run.
python ablation_study.py

# Judge the released 500 nodes and 500 edges, without resampling.
python evidence_graph_evaluation.py
```

Each entrypoint supports `--config`, `--output-dir`, and `--help`. Evaluation entrypoints also accept `--material-dir`. MathER and ablation commands accept `--query-limit`, `--query-ids`, and `--workers` for small checks; evidence evaluation accepts `--kind` and `--limit`. Subset outputs are smoke tests, not full benchmark results. Use a separate output directory for each test.

Runs resume compatible checkpoints. Changed inputs, models, or prompts must use a new output directory or the explicitly supported overwrite option. Failed API requests remain unfinished and are not converted into `NIL` predictions. The PDF pipeline checks previously completed artifacts and rejects mixing changed PDFs into an existing run.

## License

The original source code in this repository is licensed under the [Apache License 2.0](LICENSE).

Supplementary benchmark data and source materials are not covered by this code license; see their source records and licensing information. Third-party dependencies remain subject to their respective licenses.
