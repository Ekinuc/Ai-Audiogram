# 🏥 AI Clinical Audiogram Analysis System (V2)

An advanced deep learning diagnostic support tool utilizing a **6-frequency full spectrum neural network** to automate audiogram interpretation. The application processes audiogram curves (Right/Left ear), predicts exact hearing thresholds in decibels (dB HL), calculates clinical metrics (PTA, disability rates), and generates professional clinical PDF reports.

---

## 🌟 Key Features

- **Dual-Ear AI Inference:** Supports analyzing a single combined colored audiogram image (automatically splitting red/blue channels) or separate right/left ear images.
- **6-Frequency Analysis:** Detects precise hearing levels at **250 Hz, 500 Hz, 1000 Hz, 2000 Hz, 4000 Hz, and 8000 Hz**.
- **Interactive Target Marker:** Includes a slider to interpolate intermediate frequencies on a logarithmic scale with visual target indicators.
- **Morphological Diagnostics:** Automatically identifies **Classic V-Notches** (high probability indicators of Noise-Induced Hearing Loss (NIHL) or Tinnitus) and calculates clinical inter-aural asymmetry.
- **Automated Clinical Reporting:** Generates comprehensive PDF reports with tabular data, diagnostic conclusions, and visual curves.
- **High-Performance Architecture:** Uses a custom pre-trained Keras CNN model (`Model_100K_FINAL_SNIPER_H.keras`).

---

## 🛠️ Prerequisites & System Requirements

- **Python Version:** `Python 3.12` is recommended (compatible with `3.8` through `3.12`).
- **Disk Space:** ~50 MB (excluding Python packages).
- **Git LFS (Optional):** Since the model file is approximately **27.6 MB**, standard Git works fine (GitHub's single file limit is 100 MB). However, if you plan to track larger assets in the future, Git LFS can be used.

---

## 🚀 Installation & Local Setup

Follow these simple steps to set up and run the application on any computer (Windows, macOS, or Linux).

### 1. Clone or Download the Project
Download the repository files to your local machine.

### 2. Set Up a Virtual Environment (`venv`)
It is highly recommended to use a virtual environment to avoid dependency conflicts. Open your terminal/command prompt, navigate to the project directory, and run:

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
call venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
Install all the required Python packages using the provided `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Start the interactive web interface:
```bash
streamlit run app2.py
```

After running this command, your default web browser will automatically open the app at `http://localhost:8501`. If it doesn't open automatically, copy and paste the URL into your browser.

---

## 📂 Project Structure

To run the application, ensure the files are placed in the same directory:
```
audiogram-analyzer/
├── app2.py                          # Streamlit application entrypoint & logic
├── Model_100K_FINAL_SNIPER_H.keras  # Pre-trained deep learning Keras model
├── requirements.txt                 # Exact package dependencies
├── .gitignore                       # Rules to keep the repository clean
└── README.md                        # Documentation (This file)
```

---

## 💡 How to Use the App

1. **Enter Patient Details:** Input the patient's name in the sidebar for the PDF report.
2. **Choose Upload Mode:**
   - **Single Combined Image:** Upload one image containing both red (Right) and blue (Left) curves. The system splits them automatically.
   - **Separate Images:** Upload individual files for the Left and Right ears.
3. **Analyze Interactive Curves:** Use the logarithmic slider to scan and interpolate intermediate frequencies.
4. **Download Clinical Report:** Scroll down and click **"Download Clinical PDF Report"** to export a print-ready clinical report.

---

## ⚠️ Disclaimer

*This application is an AI-powered clinical decision support tool designed for educational and preliminary screening purposes. It is **not** a replacement for formal audiological testing or expert clinical diagnosis by a licensed ENT Specialist or Audiologist.*
