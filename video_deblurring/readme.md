🎥 Video Deblurring using RVRT (Recurrent Video Restoration Transformer)

This project demonstrates **video deblurring** using the **RVRT (Recurrent Video Restoration Transformer)** model.  
It handles **multiple types of blurs** — motion blur, Gaussian blur, defocus blur, and mixed distortions — and restores high-quality, sharp videos.

---

## 📘 Overview

The **RVRT model** is a state-of-the-art transformer-based approach designed for **video restoration tasks**, including **super-resolution, deblurring, and denoising**.  
In this project, RVRT is used **without fine-tuning**, applying pre-trained weights (`RVRT.pth`) to various blurred video datasets to test its generalization capability.

### Key Features:
- Handles **different types of blurs** on real-world and synthetic datasets  
- Uses **pre-trained RVRT model weights** for fast inference  
- Generates **high-quality restored videos**  
- Includes **notebook-based workflow** for reproducibility  

---

## 🧠 Model: RVRT (Recurrent Video Restoration Transformer)

RVRT processes video frames in a **recurrent fashion**, leveraging both spatial and temporal information efficiently.

- **Architecture:** Transformer + Recurrent Feature Propagation  
- **Input:** Sequence of blurred frames  
- **Output:** Sequence of restored frames  
- **Paper:** [RVRT: Recurrent Video Restoration Transformer (CVPR 2023)](https://arxiv.org/abs/2305.10072)

---

## 📂 Project Structure

📁 Video-Deblurring-RVRT
│
├── RVRT_without_finetune.ipynb # Main notebook for deblurring and evaluation
├── RVRT.pth # Pre-trained model weights
├── all outputs.zip # Contains deblurred video results
│ ├── Motion_Blur/
│ ├── Gaussian_Blur/
│ ├── Defocus_Blur/
│ └── Mixed_Blur/
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1. Clone or Download Repository
```bash
git clone https://github.com/yourusername/Video-Deblurring-RVRT.git
cd Video-Deblurring-RVRT
2. Create and Activate Environment
bash
Copy code
conda create -n rvrt_env python=3.9
conda activate rvrt_env
3. Install Dependencies
bash
Copy code
pip install torch torchvision torchaudio
pip install opencv-python tqdm numpy matplotlib
(Optional) Install Jupyter for notebook usage:

bash
Copy code
pip install jupyter
🧪 Usage
1. Run the Notebook
Open and execute the Jupyter Notebook:

bash
Copy code
jupyter notebook RVRT_without_finetune.ipynb
2. Input Video or Frames
Place your blurred video or frame sequence in the input/ folder (you can modify the path inside the notebook).

3. Output
The deblurred results will be automatically saved under:

php-template
Copy code
outputs/<blur_type>/<video_name>/
You can visualize or compare outputs directly in the notebook.

