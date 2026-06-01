# TA Verification Guide

This repository has two different validation stories:

1. `Full reproduction`
   This means starting the model server, downloading model weights, and running the fragmentation pipeline end to end. It is the strongest check, but it depends on the project container, the correct CUDA/PyTorch stack, Hugging Face access, and multi-GPU hardware.

2. `Evidence-based verification`
   This is the recommended grading path if the full hardware/software stack is unavailable. It lets you verify what code was added, what each addition does, and what checked-in artifacts show that the code executed on my machine.

If you only have a few minutes, use the short checklist at the end of this file.

## What Was Added

### 1. Measurement pipeline

Main files:

- `scripts/fragmentation_context_sweep.py`
- `scripts/run_fragmentation_pipeline.sh`
- `scripts/plotting/plot_context_vs_fragmentation.py`
- `scripts/plotting/plot_analytical_fragmentation.py`
- `sarathi-lean/sarathi/entrypoints/openai_server/protocol.py`
- `sarathi-lean/sarathi/entrypoints/openai_server/serving_completion.py`

What this code does:

- starts a chosen model server
- waits for readiness and verifies the served model name
- sends one request at a time
- uses exact prompt token counts instead of approximate text lengths
- limits generation to `max_tokens=1` so the run isolates allocator behavior
- shuts the server down gracefully so metrics are flushed to CSV
- plots both measured fragmentation and analytical predictions

Why it helped:

- without this pipeline, there was no clean way to measure fragmentation as a function of context length
- the OpenAI-compatible server change is important because the sweep depends on token-array prompts, not plain text prompts
- this made the project reproducible at the level of request manifests, CSV summaries, and figures

Key code landmarks:

- `scripts/fragmentation_context_sweep.py`
  - builds exact prompt token IDs by tiling a deterministic token pool
  - queries `/v1/models` first and clamps the sweep to the server's `max_model_len`
  - writes per-request metadata and a JSONL manifest
- `scripts/run_fragmentation_pipeline.sh`
  - maps a `--model-key` to the correct server wrapper
  - starts the server, waits for readiness, runs the sweep, waits for metrics, and generates plots
- `sarathi-lean/sarathi/entrypoints/openai_server/protocol.py`
  - extends `CompletionRequest.prompt` to allow `List[int]` and `List[List[int]]`
- `sarathi-lean/sarathi/entrypoints/openai_server/serving_completion.py`
  - parses token-array prompts and passes them to `engine.generate(..., prompt_token_ids=...)`

### 2. MLA path in vAttention/Sarathi

Main files:

- `sarathi-lean/sarathi/config.py`
- `sarathi-lean/sarathi/worker/cache_engine/vattention_init.py`
- `sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py`
- `sarathi-lean/tests/test_vattention_init_dispatch.py`
- `sarathi-lean/tests/test_config_cache_architecture.py`
- `sarathi-lean/tests/test_base_worker_mla_runtime_integration.py`

What this code does:

- recognizes MLA models as a distinct cache architecture instead of overloading the dense KV path
- defines MLA-specific cache geometry using two resident cache components:
  - `kv_latent`
  - `k_rope`
- computes MLA-specific page-buffer size and `tokens_per_page`
- exports a structured cache spec to the vAttention backend
- dispatches MLA initialization through a component-spec path instead of the legacy dense-KV init path

Why it helped:

- this is the code that makes MLA a first-class runtime/storage path in the project
- it is what allowed the project to analyze fragmentation for real MLA geometry rather than only for dense KV models

Key code landmarks:

- `sarathi-lean/sarathi/config.py`
  - `CacheArchitecture`, `MLAAttentionSpec`, `VAttentionCacheSpec`, and `VAttentionInitSpec`
  - `get_cache_component_specs(...)` returns `kv_latent` and `k_rope` for MLA
  - `get_page_buffer_token_bytes(...)` and related helpers compute MLA page geometry
- `sarathi-lean/sarathi/worker/cache_engine/vattention_init.py`
  - `dispatch_init_kvcache(...)` chooses between `legacy_dense_kv` and `component_spec`
- `sarathi-lean/tests/test_vattention_init_dispatch.py`
  - validates the MLA component payload shape and dispatch behavior
- `sarathi-lean/tests/test_config_cache_architecture.py`
  - verifies that MLA geometry is computed before cache planning
- `sarathi-lean/tests/test_base_worker_mla_runtime_integration.py`
  - exercises MLA prefill/decode behavior at the worker/runtime level

### 3. Synthetic Mistral MLA

Main files:

- `sarathi-lean/sarathi/model_executor/models/mistral_mla.py`
- `sarathi-lean/sarathi/model_executor/model_loader.py`
- `sarathi-lean/sarathi/config.py`
- `scripts/docker/start-server-mistral-nemo-12b-mla.sh`
- `sarathi-lean/tests/test_mistral_mla_conversion.py`

What this code does:

- takes a Mistral-Nemo GQA checkpoint
- rewrites its attention weights into an MLA-shaped scaffold
- runs that scaffold through the MLA runtime/cache path
- does this without claiming model quality preservation

Why it helped:

- this enabled the backbone-matched GQA-vs-MLA comparison that the project needed
- comparing Mistral GQA to synthetic Mistral MLA removes backbone size as a confounder

Important caveat:

- this path is for allocator/fragmentation study, not language-model quality
- the synthetic MLA cache layout is real, but the converted model is not intended to produce meaningful text

Key code landmarks:

- `sarathi-lean/sarathi/model_executor/models/mistral_mla.py`
  - normalizes source weights
  - builds `kv_latent_proj`, `k_rope_proj`, and `kv_up_proj`
  - loads the resulting scaffold into an MLA model class
- `sarathi-lean/sarathi/config.py`
  - `maybe_apply_mistral_mla_conversion(...)` rewrites the HF config early so cache planning uses MLA geometry
- `sarathi-lean/sarathi/model_executor/model_loader.py`
  - registers and instantiates `MistralMLAForCausalLM`
- `scripts/docker/start-server-mistral-nemo-12b-mla.sh`
  - turns the conversion on with environment variables and serves Mistral through the MLA path
- `sarathi-lean/tests/test_mistral_mla_conversion.py`
  - checks that the synthetic scaffold produces the expected projection shapes

## Checked-In Evidence That The Code Ran

The strongest checked-in evidence is under `server_plots/`. Each measured model directory includes:

- `server.log`
- `context_vs_fragmentation.png`
- `context_vs_fragmentation_summary.csv`
- `analytical_fragmentation.png`
- `analytical_fragmentation_summary.csv`

These logs show the actual server command, model name, and cache geometry observed at runtime.

Important examples:

- `server_plots/qwen-14b/server.log`
  - dense KV
  - `Tokens Per Page: 819`
  - `Page Buffer Token Bytes: 2560`
- `server_plots/mistral-nemo-12b/server.log`
  - dense KV
  - `Tokens Per Page: 4096`
  - `Page Buffer Token Bytes: 512`
- `server_plots/deepseek-v2-lite/server.log`
  - MLA
  - `Tokens Per Page: 1820`
  - `Page Buffer Token Bytes: 1152`
- `server_plots/mistral-nemo-12b-mla/server.log`
  - MLA
  - `Tokens Per Page: 5461`
  - `Page Buffer Token Bytes: 384`
  - the logged launch command also shows `VATTN_ENABLE_MISTRAL_MLA_CONVERSION=1`

Why this matters:

- the synthetic Mistral MLA run does not merely reuse the dense Mistral geometry
- the runtime logs prove that the server actually came up with `Architecture: mla` and the expected MLA page geometry

## Fast Validation Options

### Option A: Lowest-friction checks on a normal machine

These do not require the full CUDA runtime or model downloads:

```bash
python -m unittest sarathi-lean/tests/test_vattention_init_dispatch.py
python -m unittest sarathi-lean/tests/test_fragmentation_context_sweep.py
```

These verify:

- MLA component-spec initialization dispatch
- exact-length sweep helper behavior

After those tests pass, the evaluator can inspect the checked-in runtime
artifacts under `server_plots/` to confirm that the reported dense-KV, real
MLA, and synthetic MLA runs produced the logged runtime geometries described in
this guide.

### Option B: Project-container checks

If the project Docker environment already exists, the repo documents this path in `docs/running-unit-tests-in-docker.md`.

Recommended commands from that doc:

```bash
docker exec -w /workspace vattn-anodyine python -m unittest discover -s sarathi-lean/tests
docker exec -w /workspace vattn-anodyine python -m unittest sarathi-lean/tests/test_config_cache_architecture.py
```

This is the preferred test path for torch-dependent tests.

### Option C: Full pipeline reproduction

Only use this if the evaluator has the project container, the correct CUDA/PyTorch environment, Hugging Face access, and the required GPUs.

Representative commands:

```bash
scripts/run_fragmentation_pipeline.sh --model-key qwen-14b
scripts/run_fragmentation_pipeline.sh --model-key mistral-nemo-12b
scripts/run_fragmentation_pipeline.sh --model-key mistral-nemo-12b-mla
scripts/run_fragmentation_pipeline.sh --model-key deepseek-v2-lite
```

This path is not the recommended grading baseline because it is hardware-dependent.

## How To Explain The Contribution Succinctly

If you need a short grading-oriented summary, this is the core claim:

- `Pipeline:` added a deterministic end-to-end measurement pipeline that starts the server, drives exact prompt lengths, records metrics, and generates the plots used in the report.
- `MLA path:` added MLA-aware cache geometry and allocator initialization so the codebase could reason about and run MLA cache layouts instead of only dense KV layouts.
- `Synthetic MLA:` added a Mistral-to-MLA scaffold so the project could compare GQA and MLA on the same backbone, which was necessary for the main apples-to-apples architectural result.

## Suggested Grading Checklist

1. Read the design summary in `docs/final_report.tex` or the generated PDF.
2. Open `docs/Comp Arch Presentation Script - 1.md` for the intended explanation of the pipeline and synthetic MLA story.
3. Inspect the three contribution areas listed above in the code.
4. Run the lightweight unit tests if desired.
5. Inspect `server_plots/*/server.log` and the corresponding plots to confirm that the code produced runtime artifacts for dense KV, real MLA, and synthetic MLA runs.

That gives a reasonable basis to verify both:

- what code was added
- and that the added code was used to produce the reported results
