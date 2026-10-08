# N46 on the training PC (Windows, NVIDIA RTX 2060 SUPER)

This PC fine-tunes the product's detector (RF-DETR Medium, Apache-2.0) on SoccerTrack v2 boxes (CC BY 4.0) cut into
the product's own tiles, then runs the product's tiled detector on the held-out clips. The Mac does all the judging.

## 1. Set up (about 30 minutes, once)

1. Update the NVIDIA driver (GeForce Experience or nvidia.com).
2. Install **Python 3.12** from python.org (tick "Add python.exe to PATH").
3. Download every `.zip` from the GitHub release **n46-kit-1**
   (https://github.com/danrublop/Fut-Traffic/releases/tag/n46-kit-1 - sign in with the GitHub account that was given
   access): `n46_kit.zip`, `n46_dataset_1.zip`, `n46_dataset_2.zip`, `n46_clips.zip`, about 3.8 GB in all, into your
   Downloads folder (delete any older `n46_*.zip` there first - the line below unpacks every one it finds). Then open **Command Prompt** and paste this one line; it unpacks all of them into `C:\n46_pc`
   (not Desktop or Documents: OneDrive syncs those and locks checkpoint files mid-save):

```
for %f in ("%USERPROFILE%\Downloads\n46_*.zip") do tar -xf "%~f" -C C:\
```

   You should have:

```
C:\n46_pc\
  train.py  smoke.py  detect.py  detectors.py  README.md
  n46_smoke\  n46_dataset\  clips\
```

4. In the same **Command Prompt** (not PowerShell), paste one line at a time:

```
cd C:\n46_pc
python -m venv venv
venv\Scripts\python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
venv\Scripts\python -m pip install "rfdetr[train]==1.9.0" pytorch_lightning==2.6.1 opencv-python
venv\Scripts\python -c "import torch; print(torch.cuda.get_device_name(0), (torch.ones(1, device='cuda') + 1).item())"
```

The last line must print `NVIDIA GeForce RTX 2060 SUPER 2.0`. If it errors, send the message.
Versions are pinned because the Mac tested exactly these.

The first training and the first `--weights stock` run download the stock weights (internet needed once).

## 2. Smoke test (about 45 minutes)

Stop Windows sleeping or restarting under a long run: `powercfg /change standby-timeout-ac 0`, and in
Settings -> Windows Update pause updates for a week. Do not click inside the console window while it runs (that
pauses it until you press a key).

```
venv\Scripts\python smoke.py
```

**Send `C:\n46_pc\smoke_report.json` back** (attach it to a message - it is small). Epochs and batch size for the real training are fixed from it before any result
is seen. (Each `runs\smoke_b*\run_record.json` saying "scores too low" is expected: a 1-epoch model is not trained.)

## 3. Training: three seeds, same recipe (E and B from the smoke report)

```
venv\Scripts\python train.py --epochs E --batch B --seed 0
venv\Scripts\python train.py --epochs E --batch B --seed 1
venv\Scripts\python train.py --epochs E --batch B --seed 2
```

Each writes `runs\seed<N>\`. Its `run_record.json` must end with `"reading": "usable"` - if not, stop and send it back.
Interrupted (power, update, crash)? Run the same line again: it continues from where it stopped.
If an epoch takes far longer than the smoke report estimated, stop and send the record back.
If it fails with `WinError 1455` (paging file too small), add `--workers 0`.

## 4. Detection

```
venv\Scripts\python detect.py --weights stock --seqs 118576 118577 118578 128057 128058
venv\Scripts\python detect.py --weights runs\seed0 --seqs 118576
venv\Scripts\python detect.py --weights runs\seed1 --seqs 118576
venv\Scripts\python detect.py --weights runs\seed2 --seqs 118576
```

Send back the `detections\` folder and the three `runs\seed<N>\run_record.json`. The Mac picks the median seed on
118576 and commits that choice; only then, on the four test clips:

```
venv\Scripts\python detect.py --weights runs\seed0 --seqs 118577 118578 128057 128058
venv\Scripts\python detect.py --weights runs\seed1 --seqs 118577 118578 128057 128058
venv\Scripts\python detect.py --weights runs\seed2 --seqs 118577 118578 128057 128058
```

## Licences

SoccerTrack v2 data: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) - the tiles here are cropped, resized
and re-encoded from it. RF-DETR code and Medium weights: Apache-2.0. Nothing in this folder is non-commercial or AGPL.
