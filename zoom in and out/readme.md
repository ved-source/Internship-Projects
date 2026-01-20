A clear, production-ready README is below. It covers what’s inside the zip, how to set up, and how to run each component with step-by-step instructions and notes.

AI Webcam Zoom & Video Deblur Suite
Interactive Jupyter notebooks for high-magnification webcam zoom controllers (50x–500x) with optional enhancement, an AI-enhanced Keras/TensorFlow.js demo, and a PyTorch-based video deblurring script using a Video Restoration Transformer (VRT).​

Contents
Camera_Zoom_in-out.ipynb — Base webcam zoom controller up to 50x with mouse/touch support.​

Camera_Zoom_in-out_100x.ipynb — Zoom controller variant capped at 100x.​

Camera_Zoom_in-out_200x.ipynb — Zoom controller variant capped at 200x.​

Camera_Zoom_in-out_500x_enhanced.ipynb — 500x zoom with real-time enhancement at high zoom levels.​

Camera_Zoom_in-out_500x_enhanced_upto100.ipynb — 500x UI/logic with auto-enhancement; optimized experience messaging for up to 100 frames in logs.​

Image_Zoomin_Keras_CNN.ipynb — AI-enhanced webcam demo using a lightweight CNN via TensorFlow.js for on-the-fly image improvement up to 500x.​

video_deblur.py — Batch video deblurring tool using a VRT backbone, with CLI for model/video paths.​

requirements.txt — Minimal Python package list for core dependencies (extend as needed).

Features
Trackpad pinch and mouse wheel zoom, click-drag panning, double-click reset across notebooks.​

Visual zoom indicator, adaptive zoom steps, and touch support on mobile devices.​

500x variants include optional enhancement pipeline that activates automatically at high zoom (e.g., ≥20x) and can be toggled manually.​

Keras/TensorFlow.js demo overlays simulated CNN enhancement when zoom surpasses a threshold (e.g., 1.5x).​

Video deblurring script includes VRT modules, frame extraction, pre/post-processing, and MP4 writing.​

Quick Start
Prerequisites
Python 3.9+ recommended for scripts; Jupyter for notebooks.

A modern browser with camera permission for webcam notebooks.​

For video deblurring: CUDA-capable GPU recommended; PyTorch + torchvision installed; a compatible VRT .pth checkpoint file.​

Install
Create/activate a virtual environment, then:

pip install -r requirements.txt

For video_deblur.py, also install PyTorch with the right CUDA version per your system.​

For Keras CNN notebook, the environment loads TensorFlow.js in-browser; Python-side installs listed in the notebook are minimal and primarily for Jupyter helpers.​

Using the Webcam Zoom Notebooks
Open any notebook in Jupyter/Colab and run all cells:

Camera_Zoom_in-out.ipynb:

Provides zoom up to 50x with zoom-in/out buttons, mouse/touch gestures, pan, and reset.​

100x/200x variants:

Same UI/UX with increased MAXSCALE to 100 and 200 respectively.​

500x enhanced:

Adds an “Enable Enhancement” toggle; enhancement auto-activates at high zoom levels and uses canvas-based sharpening/contrast adjustments tuned to scale.​

500x enhanced (upto100):

Similar to above; includes messaging prompts and handling oriented to testing with up to 100 frames in logs/UI prompts.​

Controls:

Scroll or pinch to zoom; adaptive step sizes for smoother control at high magnification.​

Click-drag to pan when zoomed.​

Double-click to reset.​

Toggle enhancement on 500x variants for improved clarity.​

Notes:

Browser will request camera permission; ensure HTTPS or localhost when applicable.​

Enhancement applies progressive sharpening/contrast beyond certain zoom thresholds for readability.​

AI-Enhanced Webcam (Keras/TensorFlow.js)
Image_Zoomin_Keras_CNN.ipynb demonstrates a lightweight, in-browser CNN using TensorFlow.js:​

Click “Start Webcam” to init stream.​

Zoom up to 500x; a status badge turns “CNN Active” once zoom exceeds threshold (e.g., 1.5x).​

The canvas applies simulated enhancement (contrast/brightness/sharpening) each frame; code includes a minimal sequential conv stack compiled in TensorFlow.js.​

Environment notes:

The notebook attempts pip installs for common helpers, but core enhancement runs via TF.js loaded from CDN; no heavy server-side training is required.​

If attempting Real-ESRGAN pip install inside the notebook, note that it may fail in some Colab/PyPI contexts without the correct extras; it’s optional and not required for the demo.​

Video Deblurring (VRT)
The script processes an input video into a deblurred output using a VRT-like PyTorch architecture.​

Example
Prepare paths:

--model_path: Path to VRT .pth (e.g., C006VRTvideodeblurringGoPro.pth).​

--input_video: Path to blurry input MP4.​

--output_video: Desired output MP4 path.​

Run:

python video_deblur.py --model_path /path/model.pth --input_video /path/input.mp4 --output_video /path/output.mp4 --device cuda --batch_size 6 --max_frames 300​

Behavior:

Extracts frames, resizes to target size (default 256x256), normalizes to , and stacks into B,C,T,H,W for model inference.​

Postprocesses to uint8 BGR and writes MP4 with cv2.VideoWriter.​

Includes defensive checks for missing files and falls back conservatively if model is None.​

Model compatibility:

The loader filters keys and warns on mismatches; strict=False to allow close-but-not-identical checkpoints. Exact VRT variant alignment yields best results.​

Performance tips:

Use --device cuda on a GPU machine.​

Increase batch size if VRAM allows; frames are padded in batches.​

Project Structure
Notebooks use Python to inject HTML/JS into Jupyter outputs, building the webcam UI dynamically with ipywidgets and Javascript.​

Enhancement logic in 500x notebooks applies scale-aware sharpening and contrast within an offscreen canvas, driven by requestAnimationFrame.​

The Keras CNN notebook builds a small TF.js sequential model in-browser, flips into “active” mode above a zoom threshold, and applies simulated enhancement each frame.​

The deblurring module defines core VRT components (3D Swin attention blocks, patch embedding, layers) and a CLI for end-to-end processing.​

Requirements
requirements.txt includes minimal core packages; extend for your environment:

opencv-python-headless, numpy, pillow (and any additional libs used in your runtime).

For notebooks, Jupyter, ipywidgets, and a modern browser.​

For deblurring, PyTorch/torchvision with CUDA if available.​

Troubleshooting
Webcam permission errors: ensure browser permissions and secure context (HTTPS/localhost); re-run cells after permitting.​

Enhancement performance at 500x: high zoom can be GPU/CPU intensive; toggle enhancement off or reduce zoom if frames drop.​

TF.js model not loading: check CDN access; the notebook injects tf.min.js dynamically and updates status badges.​

VRT checkpoint mismatch: ensure the checkpoint matches architecture; the loader prints skipped keys and proceeds with strict=False.​

Video I/O issues: verify codec availability and paths; script uses mp4v FourCC for broad compatibility.​

License and Attribution
The provided notebooks and script are intended for educational and experimental use; verify third-party model licenses and datasets before distribution.​

If using external VRT checkpoints or Real-ESRGAN, comply with their respective licenses and citation requirements