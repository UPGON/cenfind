# Installing CenFind (fixed fork)

This is [UPGON/cenfind](https://github.com/UPGON/cenfind) — the centriole-detection
pipeline published in Bürgy, Weigert, Hatzopoulos et al., *BMC Bioinformatics* 2023 —
with a set of performance, safety, testing, and dependency fixes applied on top.

**These fixes are not on PyPI and not merged upstream.** Running `pip install cenfind`
installs the original, unmodified package (currently 0.15.2, restricted to
Python ≥3.9,<3.11). To get the fixed version, install from the fork below.

- Fork: https://github.com/ghatzop/cenfind — branch `improvements`
- Review PR (diff of every change, with CI results): https://github.com/ghatzop/cenfind/pull/1
- Not pushed to `UPGON/cenfind` — this is pending review before any upstream contribution.

## What's fixed vs. the standard release

| Fix | Why it matters |
|---|---|
| Model-reload bug | SpotNet/StarDist models were reloaded from disk on every field/channel instead of once; now cached. Faster on large batches. |
| Removed debug `__main__` block in `score.py` | Old code could `shutil.rmtree` a dataset's output folders if the file was ever run directly instead of via the `cenfind` command. |
| Fixed import-time side effects | Importing the detection module no longer mutates global random seeds; TensorFlow log suppression now actually takes effect. |
| `cenfind download-model` command | Downloads, checksums, and extracts the model weights automatically — no more manual Figshare download. |
| GitHub Actions CI | Tests now run automatically on push/PR across Python 3.9/3.10/3.11. |
| Expanded test coverage | The core nucleus-centriole assignment logic (previously untested) now has tests, along with the data structures and cilia detection. |
| Refreshed dependency pins | Python support widened from `<3.11` to `<3.12`; TensorFlow bumped from a 2022-era 2.9.0 to 2.15.1 (kept below 2.16 to avoid Keras 3's breaking changes); numba/protobuf/ortools/pandas/scipy/scikit-image bumped for Python 3.11 wheel availability. |
| Zip-slip guard + download timeout | Hardening on the new `download-model` command. |

Every change above was verified against a real multi-field dataset — detection output
(nuclei/centriole counts per field) is byte-identical to the original release.

## Known limitations

- `albumentations` is capped at `<1.4.0` because 1.4.x removed the `Flip` transform
  that `training/config.py` uses. Not fixed here — that training code path wasn't
  covered by the dataset-driven verification this pass relied on. Training may not
  work on the refreshed dependency stack until that's addressed separately.
- Verified against one real dataset (one staining/imaging condition), not the full
  diversity of conditions the lab uses. Strong evidence, not exhaustive proof.

---

## Install

### Prerequisites

- Python 3.9, 3.10, or 3.11 (check with `python3 --version`; use `pyenv` if you need
  to install a specific version)
- git

### 1. Clone the fork

```bash
git clone https://github.com/ghatzop/cenfind.git
cd cenfind
git checkout improvements
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install cenfind

```bash
pip install -e .
```

This installs cenfind in editable mode, so `git pull` on the fork picks up updates
without reinstalling.

### 4. Get the model weights

```bash
cenfind download-model /path/to/models/
```

This downloads the official weights from Figshare, verifies their checksum, and
extracts them to `/path/to/models/master`. (On this machine, the weights are already
present at `models/master` and `models/denovo` alongside this file — no need to
re-download.)

### 5. Verify the install

```bash
cenfind --help
```

Optional — run the test suite (self-contained, no external data needed):

```bash
pip install pytest
pytest tests -W ignore::DeprecationWarning
```

### 6. Run it

```bash
cenfind score /path/to/dataset /path/to/models/master -n 0 -c 1 2 3
```

- `dataset` must contain a `projections/` subfolder of `.tif` images.
- `-n` is the nuclei channel index, `-c` the centriole channel(s) to score.
- Add `--cpu` to force CPU-only inference.

Results land in `predictions/`, `visualisation/`, and `statistics/` under the
dataset folder.

---

## This machine's setup

Already configured and verified as of this handover:

| What | Where |
|---|---|
| Source (git checkout, branch `improvements`) | `/Users/hatzopou/Dropbox/MyDocs/PROJECTS/Auto_Scoring/4_cenfind` |
| Model weights | `/Users/hatzopou/Dropbox/MyDocs/PROJECTS/Auto_Scoring/4_cenfind/models/master` (and `.../models/denovo`) |
| Python environment (editable install, ready to use) | `/Users/hatzopou/Dropbox/MyDocs/Coding/cenfind/cenfind_venv` |
| Example dataset | `/Users/hatzopou/Dropbox/MyDocs/Coding/12907` |

To use it directly:

```bash
/Users/hatzopou/Dropbox/MyDocs/Coding/cenfind/cenfind_venv/bin/cenfind score \
  /path/to/dataset \
  /Users/hatzopou/Dropbox/MyDocs/PROJECTS/Auto_Scoring/4_cenfind/models/master \
  -n 0 -c 1 2 3
```

Or activate the environment first:

```bash
source /Users/hatzopou/Dropbox/MyDocs/Coding/cenfind/cenfind_venv/bin/activate
cenfind score /path/to/dataset .../models/master -n 0 -c 1 2 3
```
