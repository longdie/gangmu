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

Configure the required connection roles in `config.toml`. Each role has an API key, base URL, and protocol (`openai_chat_completions`, `claude_messages`, or `openai_responses`, depending on the task). Environment variables such as `GANGMU_LLM_API_KEY` and `GANGMU_LLM_BASE_URL` override the corresponding values. Embedding and reranking use OpenAI-compatible embedding and `/rerank` endpoints. OCR uses a chat-completions endpoint supporting image inputs. Claude Code Agent requires a separately installed `claude` executable (the paper used Claude Code v2.1.220). For its connection role, `claude_messages` connects directly to a Messages endpoint; `openai_chat_completions` uses a temporary local adapter to a Chat Completions endpoint. Both require tool-call support and a response model ID matching `[models.agent].name`. The adapter preserves the Claude Code agent, MCP tools, prompts, and output validation; it does not substitute another model or agent.

| Role | Used for | Default model |
| --- | --- | --- |
| `llm` | PDF extraction, normalization, canonical graph construction | GPT-5.6-terra |
| `llm` | Gangmu, BM25 + SEV, four ablations | GPT-5.6-terra |
| `embedding` | Vector generation | Qwen3-Embedding-8B, 2048 dimensions |
| `reranker` | Candidate reranking | Qwen3-Reranker-8B |
| `ocr` | PDF parsing | GLM-5V-Turbo |
| `judge` | Evidence Graph correctness | Claude Opus 5 |
| `agent` | Claude Code Agent baseline | GPT-5.6-terra |
| `baseline` | AutoMathKG and LLMForEM, via Responses API | GPT-5.6-terra |

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
  prompts/            # Gangmu, evaluation, and agent prompt templates
  scripts/            # pipeline and experiment implementations
  src/                # shared graph and model components
```

`Material/`, `inputs/`, `outputs/`, `experiments/`, `tmp/`, local credentials, and Python caches are ignored by Git. Supplementary data are read without modification. Relative command-line paths are resolved from the repository root, even when the command is invoked from another working directory.

## Run the four entrypoints

```bash
# Complete PDF-to-graph pipeline; includes PDFs in nested folders.
python main.py --input-dir inputs --output-dir outputs/documents

# Gangmu and all eight reported baselines on the 512 released queries.
# AutoMathKG also requires requirements-automathkg.txt (see below).
python mather_evaluation.py

# Only the four ablations; no separate Gangmu control run.
python ablation_study.py

# Judge the released 500 nodes and 500 edges, without resampling.
python evidence_graph_evaluation.py
```

Each entrypoint supports `--config`, `--output-dir`, and `--help`. Evaluation entrypoints also accept `--material-dir`. MathER and ablation commands accept `--query-limit`, `--query-ids`, and `--workers` for small checks; evidence evaluation accepts `--kind` and `--limit`. Subset outputs are smoke tests, not full benchmark results. Use a separate output directory for each test.

Runs resume compatible checkpoints. Changed inputs, models, or prompts must use a new output directory or the explicitly supported overwrite option. Failed API requests remain unfinished and are not converted into `NIL` predictions. The PDF pipeline checks previously completed artifacts and rejects mixing changed PDFs into an existing run.

## AutoMathKG and LLMForEM

Both adapters use the frozen MathER-Alg KB and the shared result exporter. Configure `[providers.baseline]` with a Responses API endpoint and credentials, or set `GANGMU_BASELINE_BASE_URL` and `GANGMU_BASELINE_API_KEY`. Model IDs are configured separately in `[models.automathkg]` and `[models.llmforem]`. The reported settings use GPT-5.6-terra with low reasoning effort, no temperature override, and output limits of 4,096 and 1,024 tokens, respectively. The service must return the configured model ID. These two adapters retain serial model calls regardless of `--workers`.

AutoMathKG additionally requires a local sentence-transformers encoder:

```bash
pip install -r requirements-automathkg.txt
python mather_evaluation.py --methods automathkg llmforem

# Small check; use a different output directory for a full run.
python mather_evaluation.py --methods automathkg llmforem \
  --query-limit 2 --output-dir tmp/published-baselines-smoke
```

`automathkg` reimplements the entity-fusion procedure from [AutoMathKG](https://arxiv.org/abs/2505.13406), preserving its published pairwise and multiple-match selection instructions. MathVD1 concatenates title, field, contents, incoming references, and outgoing references. Canonical references come from the supplied canonical edges, preserving their dependency/proof labels; query references are empty. The encoder is `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with CPU execution, batch size 32, maximum sequence length 256, and 384-dimensional vectors. Cosine retrieval returns five candidates across both node types, without a threshold or reranker. Every pair is judged; multiple affirmative matches trigger a separate selection call over those matches only. A new-entity decision becomes `NIL`, and the KB remains fixed. The first run downloads the encoder; `[automathkg].embedding_model` can instead point to a local model directory, with `local_files_only = true` for offline loading. Vector caches stay in the runtime checkpoint directory.

`llmforem` retains the [official LLMForEM zero-shot prompt](https://github.com/wbsg-uni-mannheim/MatchGPT/tree/ba1691935c744bb0eacc9abf772f39271e318e4b/LLMForEM) and its value-only serialization: type, title, and statement separated by spaces. Queries use normalized statements, and candidates use canonical statements. It retrieves 20 same-type candidates using the supplied Qwen3 vectors, judges all 20 independently, and returns the highest-ranked affirmative match or `NIL`. It does not use MiniLM or reranking. Both adapters explicitly allowlist prompt fields, excluding gold labels and targets; their original prompt strings are kept in `scripts/experiment/automathkg.py` and `scripts/experiment/llmforem.py`.

Successful pair judgments are checkpointed individually. Invalid answers, refusals, incomplete responses, and transport errors leave queries unfinished after bounded retries. Rerun the same command to resume. Results are exported to `AutoMathKG/` and `LLMForEM/` under the chosen output directory, each containing `predictions.json` and `metrics.json`.

Offline checks use synthetic inputs and mocked responses, without model downloads or API calls:

```bash
python -m unittest discover -s scripts/experiment/tests -v
```

## License

The original source code in this repository is licensed under the [Apache License 2.0](LICENSE).

Supplementary benchmark data and source materials are not covered by this code license; see their source records and licensing information. Third-party dependencies remain subject to their respective licenses.
