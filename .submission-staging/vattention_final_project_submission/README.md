# Final Project Submission

This curated bundle contains the final paper, the key implementation files for
the project pipeline and MLA work, the lightweight unit tests that can run
without the full hardware stack, and the runtime artifacts used in the report.

Public repository:

- [https://github.com/Anodyine/vattention](https://github.com/Anodyine/vattention)

## Contents

- `paper/docs/final_report.pdf`
  Final paper for submission.
- `paper/docs/grader-verification-guide.md`
  Short guide for TAs showing what code was added, where to look, and what to
  run first.
- `implementation/`
  Focused source files for:
  - the measurement pipeline
  - the MLA cache path
  - the synthetic Mistral MLA path
  - the associated unit tests
- `artifacts/server_plots/`
  Checked-in logs, plots, and CSV summaries referenced by the paper.

## First Verification Steps

From the repository root, the two lightweight tests to run are:

```bash
python -m unittest sarathi-lean/tests/test_vattention_init_dispatch.py
python -m unittest sarathi-lean/tests/test_fragmentation_context_sweep.py
```

These are the host-side tests recommended in the report appendix and the
verification guide because they do not require the full project hardware stack.

## Notes

- The full pipeline and some of the stronger MLA tests depend on the project
  container, PyTorch/CUDA, model access, and multi-GPU hardware.
- The `server_plots/` artifacts are included so the evaluation can still verify
  that the reported dense-KV, real MLA, and synthetic MLA runs were produced by
  the implementation even if the evaluator cannot reproduce the full runtime
  environment locally.
