"""Generic VLM-as-judge package.

Module layout:

* :mod:`scripts.judge.api` — async vision-judge call (reuses ``utils.vlm``).
* :mod:`scripts.judge.csv_io` — append-only CSV per judge with skip-on-prior-verdict.
* :mod:`scripts.judge.bundles` — per-judge ``Bundle`` builders (image labels,
  paths, question).
* :mod:`scripts.judge.configs` — registry of all 16 paper-scale judges.
* :mod:`scripts.judge.cli` — orchestrator for ``scripts/judge.py``.
"""
