# Dev Log: 2026-04-06

## Scope

This log covers the work done today on branch:

- `adding-analytical-predicitons`

The focus today was:

- adding analytical predictions to the fragmentation figures
- deciding how those analytical predictions should be presented
- extending the plotting and pipeline tooling so the analytical figures are reproducible
- designing long-context Mistral GQA vs synthetic Mistral MLA experiments
- debugging the synthetic MLA startup/capacity path
- dealing with thermal constraints on the Threadripper host while running long sweeps
- preparing the shorter `~8k` MLA vs MHA comparison setup

This should be treated as the authoritative handoff point for resuming tomorrow.

## Branch / Working Tree State

Current branch:

- `adding-analytical-predicitons`

Working tree at time of writing is dirty.

Important modified source files:

- `/home/anodyine/repos/vattention/sarathi-lean/sarathi/config.py`
- `/home/anodyine/repos/vattention/sarathi-lean/sarathi/engine/arg_utils.py`
- `/home/anodyine/repos/vattention/sarathi-lean/sarathi/model_executor/model_loader.py`
- `/home/anodyine/repos/vattention/sarathi-lean/tests/test_config_cache_architecture.py`
- `/home/anodyine/repos/vattention/scripts/fragmentation_context_sweep.py`
- `/home/anodyine/repos/vattention/scripts/plotting/plot_context_vs_fragmentation.py`
- `/home/anodyine/repos/vattention/scripts/run_fragmentation_pipeline.sh`

Important untracked source/doc files:

- `/home/anodyine/repos/vattention/scripts/plotting/plot_analytical_fragmentation.py`
- `/home/anodyine/repos/vattention/docs/analytical_fragmentation_predictions.tex`
- `/home/anodyine/repos/vattention/docs/proposal2.pdf`

There are also many regenerated plot artifacts under `server_plots/...` and LaTeX build artifacts under `docs/`. Do not assume the worktree is clean tomorrow.

## High-Level Outcome

By the end of today we had:

- a clean analytical story for fragmentation:
  - exact analytical sawtooth
  - worst-case smooth envelope
  - average approximation
- empirical fragmentation plots overlaid with the worst-case envelope
- a separate analytical-only plot that shows the exact sawtooth plus the worst-case envelope
- a LaTeX note in `docs/` that writes down the formulas for dense KV and MLA
- pipeline support for generating the analytical companion figure automatically
- startup snapshots of `hf_config.json`, `cache_layout.json`, and `vattention_cache_spec.json` for future reproducibility
- a regression fix for the synthetic Mistral MLA capacity-planning bug
- a regression test that verifies the synthetic Mistral MLA cache planning path produces MLA geometry (`tokens_per_page = 5461`) before cache layout is computed

The main unresolved practical issue is not the old startup geometry bug anymore. It is the actual runtime memory ceiling for very large synthetic MLA prompts. The model now starts much farther into the long-context regime, but eventually OOMs during execution around the `114k` region on the current hardware/configuration.

## Conceptual Decisions We Settled On

### 1. What analytical curve belongs on the main empirical plot?

We decided not to overlay the exact analytical sawtooth on top of the empirical sawtooth in the main figure.

Reason:

- it is correct, but visually too busy
- the exact sawtooth mostly duplicates the empirical line once the allocator is behaving as expected
- the more useful visual claim in the main figure is that the theory predicts the scale of the peaks

So the main empirical plot should use:

- the empirical sawtooth
- a dotted worst-case envelope

The exact analytical sawtooth should live in:

- a separate analytical plot, or
- a side-by-side comparison figure, if needed later

### 2. Which analytical formula should be emphasized?

We clarified the distinction between three different quantities:

1. Exact fragmentation:

`F_exact(C) = 100 * (ceil(C/T) * T - C) / (ceil(C/T) * T)`

2. Worst-case envelope:

`F_worst(C) = 100 * T / (C + T)`

3. Average approximation:

`F_avg(C) = 100 * (T/2) / (C + T/2)`

Where:

- `C` = context length in tokens
- `T` = tokens per page

For the report-quality overlay, we settled on:

- use `F_worst(C)` on the empirical plot
- keep `F_exact(C)` in the analytical companion figure and in the writeup

### 3. What is the right interpretation of the proposal’s old MHA formula?

We clarified that the original proposal intuition was close for average-case byte waste, but not for the plotted fragmentation percentage.

Important distinction:

- the proposal’s earlier model was about waste in bytes across the KV cache
- the plotted `% fragmentation` is allocator tail slack relative to mapped token capacity

So the fragmentation curves for the figures should be written in terms of:

- `tokens_per_page`
- model cache geometry
- context length

not in terms of full-cache resident bytes alone.

### 4. How should the empirical and analytical figures be made visually comparable?

We intentionally made the empirical raw-sawtooth figure and the analytical figure more symmetric:

- removed the lower block-count panel from the default raw empirical plot
- set the empirical y-axis to `0..100` with lower padding
- added left and bottom padding to the empirical plot to match the analytical figure
- restored the `KV blocks mapped` colorbar on the empirical plot
- added a horizontal red dotted line at `5%`, labeled as the significance threshold

The comparison story is now:

- main empirical plot: readable and presentation-friendly
- analytical plot: precise and theory-oriented

### 5. What fragmentation threshold do we care about?

We added a fixed red dotted line at `5%` on the empirical figure and treated it as the threshold below which fragmentation is “no longer significant.”

This is now part of the current visual language of the fragmentation plots.

## Source Changes Made Today

### 1. Empirical fragmentation plotter

File:

- `/home/anodyine/repos/vattention/scripts/plotting/plot_context_vs_fragmentation.py`

Main changes:

- restored the `KV blocks mapped` colorbar for raw empirical plots
- default raw plot is single-panel
- lower block-count step panel is now optional via `--show-block-panel`
- empirical plot now includes:
  - worst-case dotted envelope
  - `5% significance threshold`
  - y-axis fixed to `0..100` with lower padding
  - left padding and bottom padding to visually match the analytical figure
- added `--max-context` filtering so longer CSVs can be reused for shorter-window comparisons
- summary CSV includes analytical error metrics

Practical outcome:

- the same long-run metrics files can now be replotted into shorter `~8k` windows without rerunning everything

### 2. New analytical-only plotter

File:

- `/home/anodyine/repos/vattention/scripts/plotting/plot_analytical_fragmentation.py`

This script:

- builds exact analytical sawtooth curves
- optionally overlays the worst-case envelope
- infers geometry from:
  - saved `server-output` artifacts
  - `server.log`
  - explicit model arguments
  - Hugging Face config when needed

Important fixes:

- `maybe_int()` now treats `"None"` as null instead of crashing on `int("None")`
- older runs can still be reconstructed from `server.log`
- future runs can use the saved `hf_config.json` and `cache_layout.json`

### 3. Pipeline support for analytical plots

File:

- `/home/anodyine/repos/vattention/scripts/run_fragmentation_pipeline.sh`

Changes:

- analytical companion plot generation added to the pipeline
- pipeline now emits:
  - `analytical_fragmentation.png`
  - `analytical_fragmentation_summary.csv`
- pipeline now starts the server with:
  - `--replica_scheduler_max_batch_size 1`

That last change matters because these sweeps are one request at a time. Reserving for much larger active-sequence counts was unnecessary and was hurting the memory budget.

### 4. Run-time sweep pacing

File:

- `/home/anodyine/repos/vattention/scripts/fragmentation_context_sweep.py`

Changes:

- added `--inter-request-delay-seconds`
- after each request, the client can sleep before sending the next one

This was introduced specifically because the machine started approaching the Threadripper thermal ceiling during long runs.

### 5. Startup snapshotting for reproducibility

File:

- `/home/anodyine/repos/vattention/sarathi-lean/sarathi/engine/arg_utils.py`

Changes:

- engine now writes:
  - `hf_config.json`
  - `cache_layout.json`
  - `vattention_cache_spec.json`
  into `server-output/<run>/`
- `_write_json(..., default=str)` added so non-JSON-native values like dtypes do not crash startup

This was done because pulling HF config at plotting time is fragile and can drift from the actual run configuration.

### 6. Synthetic Mistral MLA startup/capacity bug fix

Files:

- `/home/anodyine/repos/vattention/sarathi-lean/sarathi/config.py`
- `/home/anodyine/repos/vattention/sarathi-lean/sarathi/model_executor/model_loader.py`

This was the most important correctness fix of the day.

#### Original bug

The synthetic Mistral GQA -> MLA conversion was happening too late.

Before today:

- the rewrite to `MistralMLAForCausalLM` happened inside `get_model(...)`
- but driver-side cache planning had already happened earlier during `ModelConfig` / engine config creation

Consequence:

- startup capacity checks still treated synthetic Mistral MLA as dense KV
- the capacity logic used `T = 4096` instead of the expected synthetic MLA `T = 5461`
- startup would reject long-context synthetic MLA runs even when the MLA geometry should have fit

Evidence that led us to this:

- failure at `121000` tokens reported:
  - `Need 30, available 23 gpu blocks`
- `ceil(121000 / 4096) = 30`
- but synthetic MLA should have needed about:
  - `ceil(121000 / 5461) = 23`

That mismatch was the smoking gun.

#### Fix

We added:

- `maybe_apply_mistral_mla_conversion(...)`

in `/home/anodyine/repos/vattention/sarathi-lean/sarathi/config.py`

and called it:

- immediately after `get_config(...)` in `ModelConfig.__init__`
- again in `model_loader.get_model(...)` so workers stay aligned

The synthetic rewrite now happens before:

- cache architecture detection
- page-buffer sizing
- tokens-per-page computation
- driver-side schedulability checks

#### Regression test

File:

- `/home/anodyine/repos/vattention/sarathi-lean/tests/test_config_cache_architecture.py`

Added test:

- synthetic Mistral MLA conversion happens before cache planning
- expected `page_buffer_token_bytes = (128 + 64) * 2`
- expected `tokens_per_page = 5461`

Verification:

- ran successfully in the container via:

```bash
docker exec vattn-anodyine bash -lc 'cd /workspace && python -m unittest sarathi-lean/tests/test_config_cache_architecture.py'
```

Result:

- `Ran 36 tests ... OK`

## Experimental Design Decisions We Settled On

### 1. Main architectural story

The project needs to show:

- fragmentation decreases as context grows
- attention architecture changes the shape and scale of that decay

The current architecture set supporting that story is:

- `Qwen-14B` as MHA
- `Mistral-Nemo-12B` as GQA
- `DeepSeek-V2-Lite` as real MLA
- synthetic `Mistral-Nemo-12B MLA` as apples-to-apples GQA vs MLA on the same backbone

This gives two complementary comparisons:

1. broad architecture comparison across different real models
2. controlled synthetic comparison on the same backbone

### 2. Short `~8k` MLA vs MHA comparison

We decided to preserve a short-window comparison around `8192` tokens to show the sawtooth shape clearly for:

- `DeepSeek-V2-Lite (MLA)`
- `Qwen-14B (MHA)`

Exact point grids inferred from the saved metrics:

#### DeepSeek `~8k` short run

```text
128, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192
```

#### Qwen `~8k` short run

```text
128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 6144, 7168, 8192
```

These runs are useful for:

- MLA vs MHA sawtooth visibility
- making it obvious that smaller `T` yields faster page turnover
- keeping the figure readable

### 3. Long Mistral GQA vs synthetic Mistral MLA comparison

This is now the main long-context design for demonstrating large-context decay.

Goal:

- push into the `100k+` range
- show how fragmentation becomes negligible at long context
- compare GQA and MLA on the same backbone

The key experimental design insight we settled on was:

- for long runs, sampling only `nT - 1` and `nT` makes the empirical curve collapse visually onto the envelope
- we need some horizontal width around the page boundary if we want the sawtooth to remain visible

We iterated several point-list designs:

1. exact best/worst pairs (`nT-1`, `nT`)
2. best/worst plus interior points
3. “about 20 before the boundary” plus the boundary (`nT-20`, `nT`)

The third option gave the most visually useful compromise for very long runs.

#### Settled GQA long-run style

For Mistral GQA (`T = 4096`), a representative long-run command used:

```text
4085,4096,8181,8192,12277,12288,16373,16384,20469,20480,24565,24576,28651,28672,32747,32768,36843,36864,40939,40960,45035,45056,49131,49152,53227,53248,57323,57344,61419,61440,65505,65536,69601,69632,73697,73728,77793,77824,81889,81920,85865,86016,90071,90112,94160,94208,98203,98304,102299,102400,106475,106496,110571,110592,114487,114688,118583,118784,122679,122880
```

This is “about 20 before each boundary” plus the boundary itself, pushed well into the `100k+` region.

#### Settled synthetic MLA long-run style

For synthetic Mistral MLA (`T = 5461`), the analogous point pattern is:

- `nT - 20`
- `nT`

Representative point list used:

```text
5441,5461,10902,10922,16363,16383,21824,21844,27285,27305,32746,32766,38207,38227,43668,43688,49129,49149,54590,54610,60051,60071,65512,65532,70973,70993,76434,76454,81895,81915,87356,87376,92817,92837,98278,98298,103739,103759,109200,109220,114661,114681
```

We intentionally removed the last four entries at one point to avoid riding too close to the cap:

- that was a design choice to reduce risk near the memory limit

### 4. Thermal pacing policy

We discovered that very long uninterrupted sweeps act as a meaningful practical thermal stress test on the machine.

Observations:

- GPUs stayed roughly in a healthy range
- the Threadripper CPU would approach the low/mid `90 C` range under sustained combined CPU+GPU load
- with no delay, we hit a dangerous region:
  - `Tctl` near `94.5 C`
  - one CCD peak at `95.75 C`

Conclusion:

- the system cools down quickly when load pauses
- the problem is not only CPU heat density
- it appears to be whole-system thermal saturation under combined sustained load

We therefore settled on:

- using `--inter-request-delay-seconds 30` for the hotter long runs
- preferring fewer, more intentionally chosen points over brute-force dense sweeps at `100k+`
- avoiding unattended long runs until cooling is improved or remounted

## Runtime / Thermal Challenges and How We Handled Them

### 1. DeepSeek startup failed because JSON export could not serialize dtype

Symptom:

- server startup crashed while writing `hf_config.json`

Cause:

- `json.dump()` encountered a dtype object

Fix:

- changed startup JSON writing to use `default=str` in `arg_utils.py`

Result:

- startup snapshots now work

### 2. Pipeline stopped before plotting when a late request failed

Symptom:

- long run would complete almost entirely
- final request would fail
- pipeline exited before graceful shutdown / plotting because of `set -e`

Impact:

- looked like all work was lost

Resolution:

- in several cases the successful rows were already present in `sequence_metrics.csv`
- we manually regenerated empirical and analytical plots from the saved metrics rather than rerunning the whole sweep

Important lesson for tomorrow:

- before rerunning a failed long sweep, inspect `sequence_metrics.csv`
- if the last row is partial but earlier rows are valid, replot from the CSV instead of starting over

### 3. Long Mistral GQA run failed on the last point

We saw this on a `131072`-cap Mistral GQA sweep:

- everything up through `114688` succeeded
- the final larger request failed

This was tolerable because:

- the successful rows were already flushed
- the empirical and analytical figures were salvageable by plotting only the completed rows

### 4. Synthetic MLA startup kept failing even after max-batch-size cleanup

This was the big debugging effort of the day.

At first it looked like:

- not enough available memory to schedule the requested max length

Reducing:

- `--replica_scheduler_max_batch_size` to `1`

was necessary but not sufficient.

The real issue turned out to be:

- driver-side cache geometry still being dense-KV because the synthetic MLA rewrite happened too late

After the fix, synthetic MLA startup progressed correctly into the long-context regime.

### 5. Synthetic MLA later failed at runtime around `114661`

After the startup bug was fixed, the synthetic MLA server finally ran deep into the long sweep.

Then we hit a different failure:

- runtime CUDA OOM during the forward pass
- failure occurred in the MLP path, not the startup cache-schedulability check

From the log:

- allocation attempt: `1.09 GiB`
- GPU 0 free memory at failure: about `1.09 GiB`
- failing prompt:
  - `prompt_chunk_len = 114661`

This is important because it means:

- the synthetic MLA geometry bug is no longer the main blocker
- the new limit is actual execution-time VRAM, not just admission planning

Practical safe region from this run:

- requests up through `109220` completed successfully
- failure happened on the subsequent request

### 6. Threadripper thermal margin was too narrow for carefree unattended long runs

Key conclusion:

- cooling is functional
- but continuous combined CPU+GPU load pushes the machine close to the CPU thermal limit

The user correctly inferred that:

- the CPU cools down quickly when demand pauses
- cooling “isn’t keeping up” when it has to reject combined CPU+GPU heat continuously

Operational conclusion we agreed on:

- a remount with fresh thermal paste is reasonable before long unattended runs
- especially because this is a Threadripper, where contact quality matters more due to the large IHS

## Current Best Understanding of the Machine Topology

From `nvidia-smi topo -m`:

- `GPU0 <-> GPU1` are linked via `NV4`
- `GPU2 <-> GPU3` are linked via `NV4`
- cross-pair traffic is `NODE`, not NVLink

Meaning:

- TP=4 uses some NVLink paths, but not an all-to-all NVLink island
- communication across the two GPU pairs is slower than within a pair

We discussed `TP=2, PP=2`, but did not adopt it for the main comparison because:

- it changes cache geometry
- that would change the fragmentation math and confound the architectural comparison

So the main comparative story should keep the parallel setup fixed rather than mixing architectural and topology changes.

## Commands / Workflows That Were Settled On

### 1. Replot a salvaged empirical figure from saved metrics

Example:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
/home/anodyine/repos/vattention/.venv-londy/bin/python \
/home/anodyine/repos/vattention/scripts/plotting/plot_context_vs_fragmentation.py \
  --input /home/anodyine/repos/vattention/server-output/mistral-nemo-12b/sequence_metrics.csv \
  --out-plot /home/anodyine/repos/vattention/server_plots/mistral-nemo-12b/context_vs_fragmentation.png \
  --out-summary /home/anodyine/repos/vattention/server_plots/mistral-nemo-12b/context_vs_fragmentation_summary.csv \
  --title "Mistral-Nemo-12B (GQA): Context Length vs Fragmentation"
```

### 2. Replot analytical companion figure from saved run geometry

Example:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
/home/anodyine/repos/vattention/.venv-londy/bin/python \
/home/anodyine/repos/vattention/scripts/plotting/plot_analytical_fragmentation.py \
  --run-dir /home/anodyine/repos/vattention/server-output/mistral-nemo-12b \
  --server-log /home/anodyine/repos/vattention/server_plots/mistral-nemo-12b/server.log \
  --out-plot /home/anodyine/repos/vattention/server_plots/mistral-nemo-12b/analytical_fragmentation.png \
  --out-summary /home/anodyine/repos/vattention/server_plots/mistral-nemo-12b/analytical_fragmentation_summary.csv \
  --title "mistralai/Mistral-Nemo-Base-2407 Analytical Fragmentation" \
  --include-worst-case
```

### 3. Short DeepSeek MLA vs Qwen MHA pipeline runs

DeepSeek short `~8k` run:

```bash
VATTN_MODEL_MAX_MODEL_LEN=8192 \
/home/anodyine/repos/vattention/scripts/run_fragmentation_pipeline.sh \
  --model-key deepseek-v2-lite \
  --context-lengths 128,512,1024,1536,2048,2560,3072,3584,4096,4608,5120,5632,6144,6656,7168,7680,8192
```

Qwen short `~8k` run:

```bash
VATTN_MODEL_MAX_MODEL_LEN=8192 \
/home/anodyine/repos/vattention/scripts/run_fragmentation_pipeline.sh \
  --model-key qwen-14b \
  --context-lengths 128,256,384,512,640,768,896,1024,1280,1408,1536,1664,1792,1920,2048,2304,2560,2816,3072,3328,3584,3840,4096,4352,4608,4864,5120,6144,7168,8192
```

### 4. Four-model cache growth comparison

Script:

- `/home/anodyine/repos/vattention/scripts/plotting/plot_cache_bytes_comparison.py`

Representative command:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
/home/anodyine/repos/vattention/.venv-londy/bin/python \
/home/anodyine/repos/vattention/scripts/plotting/plot_cache_bytes_comparison.py \
  --series '/home/anodyine/repos/vattention/server-output/qwen-14b/sequence_metrics.csv|/home/anodyine/repos/vattention/server-output/qwen-14b/benchmark_config.yml|Qwen-14B (MHA)|#d73a49' \
  --series '/home/anodyine/repos/vattention/server-output/mistral-nemo-12b/sequence_metrics.csv|/home/anodyine/repos/vattention/server-output/mistral-nemo-12b/benchmark_config.yml|Mistral-Nemo-12B (GQA)|#1f6feb' \
  --series '/home/anodyine/repos/vattention/server-output/mistral-nemo-12b-mla/sequence_metrics.csv|/home/anodyine/repos/vattention/server-output/mistral-nemo-12b-mla/benchmark_config.yml|Mistral-Nemo-12B (Synthetic MLA)|#2da44e' \
  --series '/home/anodyine/repos/vattention/server-output/deepseek-v2-lite/sequence_metrics.csv|/home/anodyine/repos/vattention/server-output/deepseek-v2-lite/benchmark_config.yml|DeepSeek-V2-Lite (MLA)|#8250df' \
  --out-plot '/home/anodyine/repos/vattention/server_plots/comparisons/four-models/cache_bytes_vs_context.png' \
  --out-summary '/home/anodyine/repos/vattention/server_plots/comparisons/four-models/cache_bytes_vs_context_summary.csv' \
  --title 'Allocated Cache vs Context Length Across MHA, GQA, and MLA Runs'
```

This is the script to use if the goal is to show that:

- MHA and DeepSeek grow faster
- Mistral GQA and especially synthetic Mistral MLA grow more slowly

## Important Artifacts Generated / Updated Today

Source / docs:

- `/home/anodyine/repos/vattention/scripts/plotting/plot_analytical_fragmentation.py`
- `/home/anodyine/repos/vattention/docs/analytical_fragmentation_predictions.tex`

Representative plot outputs:

- `/home/anodyine/repos/vattention/server_plots/mistral-nemo-12b/context_vs_fragmentation.png`
- `/home/anodyine/repos/vattention/server_plots/mistral-nemo-12b/analytical_fragmentation.png`
- `/home/anodyine/repos/vattention/server_plots/mistral-nemo-12b-mla/context_vs_fragmentation.png`
- `/home/anodyine/repos/vattention/server_plots/mistral-nemo-12b-mla/analytical_fragmentation.png`
- `/home/anodyine/repos/vattention/server_plots/deepseek-v2-lite/context_vs_fragmentation.png`
- `/home/anodyine/repos/vattention/server_plots/deepseek-v2-lite/analytical_fragmentation.png`
- `/home/anodyine/repos/vattention/server_plots/deepseek-v2-lite-8192-max-context/context_vs_fragmentation.png`
- `/home/anodyine/repos/vattention/server_plots/qwen-14b-8192-max-context/context_vs_fragmentation.png`
- `/home/anodyine/repos/vattention/server_plots/comparisons/four-models/cache_bytes_vs_context.png`

## What Is Verified vs Not Yet Verified

Verified:

- `ModelConfig` synthetic-MLA cache-planning regression test passes in-container
- empirical plotter updates are present in source
- analytical plotter exists and works on previously saved runs
- startup snapshotting works without dtype JSON crashes
- pipeline now passes `--replica_scheduler_max_batch_size 1`
- synthetic MLA startup bug is fixed enough that the run can now proceed deep into the long-context sweep

Not yet fully verified:

- a full end-to-end synthetic MLA pipeline rerun after the fix that also completes all desired high-context points
- a clean final side-by-side empirical vs analytical figure for all target runs
- whether we want to patch the multi-model comparison script to add `--max-context` filtering for shorter-window comparison plots

## Recommendations / Next Steps for Tomorrow

### 1. Use the log to avoid repeating already-solved debugging

The key solved bug is:

- synthetic MLA conversion must happen before cache planning

Do not re-debug the old `Need 30 / available 23` mismatch unless new evidence appears.

### 2. Continue from the practical synthetic MLA runtime ceiling

Treat the new problem as:

- execution-time VRAM limit around `114661`

Likely next move:

- run a slightly shorter synthetic MLA long-context sweep that stops safely below that region
- use the successful run for the GQA vs MLA long-context comparison

### 3. Preserve the short `~8k` DeepSeek vs Qwen comparison

That figure is still useful because it gives a compact MLA vs MHA visual story that is much easier to read than the `100k+` plots.

### 4. Consider patching `plot_cache_bytes_comparison.py`

Current limitation:

- no `--max-context` / `--min-context`

Potential improvement:

- add filtering so we can generate:
  - a short-window four-model cache-growth comparison
  - a long-window four-model cache-growth comparison
  from the same saved metrics files

### 5. Before unattended long runs, improve thermal confidence

We are close enough to the Threadripper thermal ceiling that it would be wise to:

- improve cooling margin
- possibly remount with fresh paste
- or at minimum continue using paced runs with long inter-request delays

## Final State Summary

The project is in a much better place tonight than it was this morning.

We now have:

- a strong analytical formulation
- better figures
- a reproducible analytical plotting path
- a clearer understanding of how to sample long-context sawtooth behavior
- a fixed synthetic MLA cache-planning path
- a tested explanation for why the synthetic MLA run now fails later and differently than before

Tomorrow should start from:

- using the successful parts of the current runs, not rerunning blindly
- treating `~109220` as the safe neighborhood reached by the synthetic MLA long run
- deciding whether the next goal is:
  - finalizing figures from existing data, or
  - running one more carefully bounded synthetic MLA long-context sweep below the new runtime OOM threshold
