<h1><img src="docs/img/logo.svg" width="56" align="absmiddle" alt=""> Fut Traffic</h1>

Working through **panoramic football film**: one fixed, wide camera that sees the whole pitch.
Video goes in. The engine calibrates the camera to the pitch, detects and tracks players and the ball,
and measures distance, speed and sprints. A local app shows the tracks, the numbers and the film.

This repository is about that one kind of footage. Panning, ball-following and broadcast
cameras are out of scope: a moving camera does not see every player, and much of what is built
here assumes a camera that stays still.

## The problems, and where each one stands

Most of the work has been finding out *why* a number is wrong, and many first answers turned out to be
wrong too. Each experiment is written up with its pre-registered pass rule and its verdict, including
the ones that failed. Withdrawn and refuted results are kept in
[the claims ledger](docs/ledgers/ledger-claims.md), together with what refuted them.

| problem | where it stands | read |
|---|---|---|
| **Mapping a stitched panorama to pitch metres.** One homography does not fit a stitched panorama, and the far touchline is hardest. | Zero-click calibration works on one install. Much of the far-side error traced back to calibration points clicked off the painted lines, so the camera is now registered to the lines themselves. A second install is needed before any of this generalises. | [n31](docs/experiments/n31-line-registration.md), [n34](docs/experiments/n34-zero-click.md), [n36](docs/experiments/n36-far-touchline.md), [n37](docs/experiments/n37-feet-on-lines.md), [n54](docs/experiments/n54-near-profile.md) |
| **Seeing players who overlap or stand far away.** | Broken tracks come from detection under occlusion, not from the linking step. Far-field tiling finds more players, and it also finds the bench. A filter removes the bench, but ID switches still block shipping it. | [n40](docs/experiments/n40-association.md), [n43](docs/experiments/n43-far-tiles.md), [n43b](docs/experiments/n43b-far-fixes.md), [n47](docs/experiments/n47-bench.md) |
| **Tracking quality against a public benchmark.** | On the one match tested, most of the gap to the best published tracker comes from the detector, not the tracker. | [n45](docs/experiments/n45-winner-boxes.md), [n41](docs/experiments/n41-conf.md), [n42](docs/experiments/n42-merge.md) |
| **Keeping a player's identity.** | Generic appearance encoders do not separate players on this footage. A sports-trained re-ID model does, but it barely moves end-to-end tracking. Jersey-number reading is unreliable at this resolution. | [st2 input ledger](docs/experiments/st2-input-ledger.md), [n49 jersey](docs/experiments/n49-jersey-ocr.md), [identity](docs/reference/identity.md) |
| **Distance and speed.** | Errors in where a player is placed lengthen their path. The calibration map is a measured lever on speed over-reads, but there is a floor that even true positions do not get under. A per-clip top-speed check cannot see the over-read because tracks are short. | [n22](docs/experiments/n22-what-inflates-distance.md), [n23](docs/experiments/n23-detector-or-map.md), [n26](docs/experiments/n26-map-oracle.md), [n27](docs/experiments/n27-tps-in-product.md), [n28](docs/experiments/n28-plausibility.md) |
| **The ball.** It is small, often occluded, and easy to confuse with clutter. | Every "pick a better path" change has lost so far. A cleaner detector exists but is not wired in, and earlier ball numbers were measured at the wrong input resolution. | [n52](docs/experiments/n52-clean-ball-data.md), [n55](docs/experiments/n55-picker-is-clutter-not-signal.md), [n65](docs/experiments/n65-every-ball-number-was-measured-at-384.md), [n67](docs/experiments/n67-fewer-candidates-does-not-help-the-picker.md) |

Current numbers live in the [ledgers](docs/ledgers/), not in this README. Each number there carries
the command that produced it and what was held fixed (the clip, the split and the threshold). Re-run
it before quoting it.

## How numbers are kept honest

Every number in the docs carries a tag. `@M` means measured, with the shell command that
re-measures it. `@C` means cited from an external source. `@D` means derived, with the arithmetic
written out. The convention is defined in [`model/validation/verify_claims.py`](model/validation/verify_claims.py):

```bash
.venv/bin/python model/validation/lint_truth.py       # mechanical checks on every doc
.venv/bin/python model/validation/verify_claims.py    # re-runs every @M command
```

`@M` commands need the dataset below. Without it they fail; they do not pass.

## Data

No full match video ships with this repository; the test fixtures include a short excerpt. The footage and ground truth come from
[SoccerTrack v2](https://github.com/AtomScott/SoccerTrack-v2) (CC BY 4.0), with ball clips from
[TeamTrack](https://github.com/AtomScott/TeamTrack) (MIT). Download it, then point the scripts at it:

```bash
export ST2_DIR=~/Downloads/st2          # default if unset
export FUTTRAFFIC_DATA=$PWD/data        # where the app keeps jobs
```

Small derived fixtures (calibrations, track excerpts, ball candidates) are committed under
`model/validation/` and are credited in [NOTICE](NOTICE). Some experiments describe models
trained on datasets whose terms forbid redistribution. Those weights are not included.

## Layout

| path | what |
|---|---|
| `model/engine/` | calibration (`autocal.py`, `calibrators.py`, `pitch.py`), detection (`detectors.py`), tracking (`track.py`, `tracking.py`, `stitch.py`), measurement (`stats.py`) |
| `model/validation/` | checks, harnesses, experiment scripts and the claim tooling |
| `app/server/` | the API and job worker (`app.py`, `jobs.py`) |
| `app/frontend/` | the UI (Next.js) |
| `app/desktop/` | an Electron shell that bundles both, with a synthetic demo match |
| `docs/` | write-ups; start at [docs/README.md](docs/README.md) or the generated [docs/INDEX.md](docs/INDEX.md) |

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r app/server/requirements.txt
.venv/bin/python app/server/app.py              # API, http://localhost:8000
cd app/frontend && npm install && npm run dev   # UI
```

The engine alone, on one video:

```bash
.venv/bin/python model/engine/autocal.py VIDEO   # zero-click calibration report for a panorama
.venv/bin/python model/engine/track.py VIDEO --calib calibration.json -o tracks.json
```

`track.py`'s command-line defaults are not the app's defaults. The configuration the app runs is
documented in [docs/reference/detection-tracking.md](docs/reference/detection-tracking.md#config).

## Licence

Code: [Apache-2.0](LICENSE). Dataset attributions: [NOTICE](NOTICE). Software dependencies:
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).
