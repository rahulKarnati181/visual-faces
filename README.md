# Webcam Expression Character

A minimal OpenCV app that shows a mirrored webcam feed beside the matching
hand-drawn neutral, happy, surprised, or sad character.

## Setup (macOS)

Python 3.10 is required. From this folder, run:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python main.py
```

On the first run, `hsemotion-onnx` downloads the
`enet_b0_8_best_afew.onnx` pretrained model to `~/.hsemotion/`. An internet
connection is needed for that one-time download. macOS may ask for camera
permission; allow it for the app launching Python (normally Terminal or Codex).

Press **Q** or close the OpenCV window to exit.
