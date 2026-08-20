# Migrating Cenfind's centriole detector to Spotiflow

## Why

Cenfind's centriole detector (`SpotNet`, from the `spotipy-detector` PyPI package) has had exactly one release, ever, and shows no sign of further maintenance. Its authors (the Weigert lab) have since published its successor, **Spotiflow** — PyTorch-based, subpixel-accurate, peer-reviewed in *Nature Methods* (2025), actively maintained, and supporting Python 3.9–3.14. Cenfind's own docs (`docs/source/contribute.rst`) already recommend this migration. This document is the concrete plan: what it requires, and the order to do it in.

Nothing here has been implemented yet — this is a planning document to align on before starting.

## Requirements

### Software / environment

- PyTorch (`pip install torch`, then `pip install spotiflow`). No conflicts expected with cenfind's existing numpy/opencv/scikit-image/pandas stack, but — following the discipline this whole effort has used — verify by actually installing into an isolated venv, not by reading version ranges.
- A likely side benefit: PyTorch's MPS backend supports Apple Silicon GPU natively, which may simplify GPU support on Mac compared to the separate `tensorflow-metal` plugin cenfind currently needs.

### Data

Spotiflow expects a specific on-disk layout, different from cenfind's current annotation format:

```
spots_data/
├── train/
│   ├── img_001.tif
│   ├── img_001.csv      # columns: y,x (subpixel float; axis-0/axis-1 also accepted)
│   └── ...
└── val/
    ├── val_001.tif
    ├── val_001.csv
    └── ...
```

Cenfind currently stores annotations as `<field>_C<channel>.txt`, comma-separated `x,y` integers (see `core/loading.py: load_foci`). The conversion is mechanical (swap column order, cast to float, restructure into train/val folders, rename extension) but is real work — a small converter script is the concrete first deliverable, not something to hand-wave.

- Fine-tune from the pretrained `"general"` model rather than training from scratch — Spotiflow documents and supports this workflow directly. How much fine-tuning data is actually needed depends on how close centriole puncta are to what `"general"` was trained on; that's unknown until tested (see Phase 1 below).
- Cenfind's own docs already define five "standard datasets" (DS1–5) as the benchmark suite for any model change. Any Spotiflow-based model must be evaluated against these — and against the current SpotNet model, side by side — before being trusted. This is now actually possible to do properly, since `cenfind evaluate` (wired up separately, see below) exists as a real command instead of a dead script.

### Engineering (inside cenfind)

- A new adapter function, parallel to `extract_foci`, that wraps `Spotiflow.from_pretrained()`/`.from_folder()` + `model.predict()` and returns cenfind's existing `Centriole` objects. This is the only real integration surface — nucleus detection, assignment, visualisation, and statistics all operate on `Centriole` objects regardless of which model produced them, so none of that needs to change.
- Model loading needs the same module-scope caching treatment already applied to `_load_foci_model`/`_load_nuclei_model` (the reload-per-call bug fixed in the 0.16.0 release) — easy to get wrong again with a new model API if not deliberate about it.
- Recommend keeping SpotNet available as a fallback (similar to how `detectors_other.py` already keeps two classical baselines pluggable) rather than a hard cutover, until Spotiflow is proven at parity or better on the standard datasets.

## Step-by-step plan

**Phase 0 — Unblock (parallel, not a gate on anything below)**
1. Resolve annotation-tool access (Labelbox recovery, or a replacement like CVAT/Label Studio). Needed for more training data regardless of which detector you end up using.

**Phase 1 — Feasibility spike (days, not weeks)**
2. `pip install spotiflow` in an isolated venv. Run `Spotiflow.from_pretrained("general")` directly against a handful of real centriole-channel images — no fine-tuning yet. This alone is informative: it tells you how far off a generic pretrained model is before investing in data conversion or fine-tuning.
3. Write the annotation-format converter (cenfind `.txt` → Spotiflow `.csv` + train/val folder layout). Needed for Phase 2 regardless of what Phase 1 shows.

**Phase 2 — Fine-tune and evaluate (the real work, and the actual go/no-go gate)**
4. Convert existing annotated data (the standard datasets plus whatever else is annotated) using the Phase 1 converter.
5. Fine-tune from `"general"` via `Spotiflow().fit(train_imgs, train_spots, val_imgs, val_spots, save_dir=...)`. Start small — one dataset, a few epochs — to validate the pipeline runs end to end before committing a full training budget.
6. Build the cenfind-side adapter (`predict()` → `Centriole` objects) in parallel with fine-tuning — it doesn't depend on final weights, just the API shape.
7. Run `cenfind evaluate` with the new adapter against the standard datasets, side by side with current SpotNet F1 scores. Apply the same threshold already documented in `retraining.rst` (F1 ≥ 0.9 use it, 0.5–0.9 marginal, < 0.5 reconsider) — just comparing two model candidates instead of "does the model still work on new data."

**Phase 3 — Ship it (only if Phase 2 clears the bar)**
8. Expose the detector choice as an option (e.g. `--detector spotnet|spotiflow`), defaulting to whichever wins the Phase 2 comparison; keep the other available for at least one release cycle.
9. Update `docs/source/retraining.rst` and `inference.rst` for the new data format and workflow.
10. Normal release process — version bump, tag, PR, merge — same mechanics already exercised for 0.16.0.

## Rough effort shape

- Phase 0–1: days — mostly waiting on annotation-tool access and running a short feasibility script.
- Phase 2: the real cost, and hard to estimate precisely — depends entirely on annotated-data volume. Likely 1–3 weeks of calendar time (annotation + training runs + evaluation), not continuous engineering effort.
- Phase 3: comparable scope to the 0.16.0 release work — a day or two of engineering once Phase 2 has a clear winner.

## Open questions to resolve during Phase 1, not before

- How close is Spotiflow's `"general"` pretrained model to centriole-like small round puncta? Unknown until tested — this determines how much fine-tuning data is actually needed.
- Does `model.predict()` return a per-point confidence/probability score usable the same way `prob_threshold` is used today? Not confirmed from documentation alone — check the actual `details` object Spotiflow returns.
