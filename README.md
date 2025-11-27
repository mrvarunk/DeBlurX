# DeBlurX: Image Reconstruction using Deep Learning


### 🚀 Project Overview
**DeBlurX** is a deep learning framework designed to restore sharp, high-frequency details from blurred images. Unlike traditional methods, DeBlurX utilizes a **Hybrid Architecture** that combines:
1.  **ResNet-UNet Backbone:** For spatial feature extraction.
2.  **FFT-Attention Blocks:** For frequency-domain restoration (recovering edges and sharpness).

### 👥 Team Members
* **Varun K** (1MS23IS140)
* **Vishal D Dhamu** (1MS23IS149)

---

### 🛠️ Tech Stack
* **Framework:** PyTorch
* **Interface:** Streamlit (Web UI)
* **Dataset:** GoPro Large Dataset
* **Model:** Custom Hybrid ViT-CNN

### ⚡ How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mrvarunk/DeBlurX.git](https://github.com/mrvarunk/DeBlurX.git)
    cd DeBlurX
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App:**
    ```bash
    streamlit run app_stylish.py
    ```

---
*Built for ISE553 Computer Vision Component.*
