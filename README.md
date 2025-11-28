

# 🎙️ Real-Time ASR Comparison System

### *CNN-LSTM vs Wav2Vec2 vs Gemini (VTU Final-Year Project)*

This project is a **real-time speech recognition comparison framework** built for academic evaluation, research demonstrations, and VTU final-year viva.
It records live audio in the browser, sends it to a FastAPI backend, and compares **three different ASR (Automatic Speech Recognition)** approaches side-by-side:

* **CNN-LSTM Baseline** (local; Vosk/custom module)
* **Wav2Vec2 Transformer Model** (Hugging Face)
* **Gemini Cloud-Style Model** (simulated inference + latency)

The system measures **transcription quality**, **latency**, and stores results in a research-style format (tables, figures, CSV logs).

---

## 🚀 Key Features

* 🎤 **Record audio directly in the browser** (MediaRecorder API)
* 🔁 **Compare 3 ASR systems on the same audio input**
* ⏱️ **Latency measurement** for each model
* 🗃️ **Automatic storage** of:

  * Recorded audio files (`results/demo_*.wav`)
  * Research-style CSV tables (WER, latency, model output)
  * Evaluation figures (`PNG` graphs)
* 📊 **Graph generation** for latency, WER, speed-comparison
* 🧪 **VTU final-year report friendly output** (tables + figures auto-generated)
* 🌐 Lightweight front-end (HTML + JS)
* ⚡ FastAPI backend with modular ASR pipeline

---

## 🧠 ASR Models Used

### 1. **CNN-LSTM Baseline (Local)**

A lightweight model ideal for offline inference.
Implemented via:

* Vosk recognizer, or
* Custom CNN-LSTM wrapper using MFCC + heuristics

Used as the “classical baseline” for comparison.

---

### 2. **Wav2Vec2 Transformer (Hugging Face)**

A self-supervised transformer model with state-of-the-art accuracy.

Model used:

* `facebook/wav2vec2-base-960h` (automatically downloaded)

---

### 3. **Gemini Cloud-Style Baseline (Simulated)**

To emulate commercial cloud ASR behaviour:

* Adds configurable artificial latency
* Produces slightly varied output for comparison
* Useful for “edge vs cloud” discussions

---

## 🏗️ Tech Stack

| Layer                | Tools Used                                         |
| -------------------- | -------------------------------------------------- |
| **Frontend**         | HTML, CSS, JavaScript, MediaRecorder API           |
| **Backend**          | FastAPI, Uvicorn                                   |
| **Models**           | Vosk / PyTorch CNN-LSTM, Hugging Face Transformers |
| **Audio Processing** | Librosa, SoundFile, pydub                          |
| **Evaluation**       | pandas, matplotlib                                 |
| **Storage**          | Local file system                                  |

---

## 📁 Project Structure

```
.
├── app.py                     # FastAPI backend
├── models/                    # ASR model files
│   └── <cnn_lstm_or_vosk_model_here>
├── templates/
│   └── index.html             # UI
├── static/
│   └── main.js                # Audio recording logic
└── results/
    ├── demo_*.wav
    ├── paper_table_*.csv
    └── figures/
        └── *.png
```
cd $HOME\Desktop
mkdir ASR_WEB_FINAL_FINAL
cd ASR_WEB_FINAL_FINAL

mkdir models
mkdir templates
mkdir static
mkdir results
mkdir results\figures

Create empty starter files
ni app.py -ItemType File
ni templates\index.html -ItemType File
ni static\main.js -ItemType File

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Typical packages include**:
`fastapi`, `uvicorn`, `torch`, `transformers`, `librosa`, `soundfile`, `pydub`, `matplotlib`, `pandas`, `vosk`.

### 4. Add Model Files

* Wav2Vec2 downloads automatically
* Place CNN-LSTM/Vosk model directory inside `/models`

Update path in `app.py` if required.

---

## ▶️ Running the Application

Start the backend:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open:

```
http://localhost:8000
```

---

## 🧪 How to Use

1. Click **Start Recording**
2. Speak your sentence
3. Click **Stop & Compare**
4. View results for:

   * CNN-LSTM
   * Wav2Vec2
   * Gemini (simulated cloud)
5. Check `/results` folder for:

   * Audio file
   * CSV “paper table”
   * Generated figures

---

## 📊 Evaluation & Graph Generation

The backend generates research-ready evaluation material:

### ✔️ Latency Bar Charts

Compare speed of all 3 ASR models.

### ✔️ WER (Word Error Rate)

Computed if a reference transcript is provided.
Otherwise, placeholders are generated (useful for report formatting).

### ✔️ Combined Summary Table (CSV)

Includes:

* Model Name
* Transcript
* Latency (ms)
* WER
* Timestamp

These are ideal for direct insertion into VTU project reports.

---

## 📚 Academic & Research Notes

* This project **does not train** models; it focuses on *comparison & system behavior*.
* All pre-trained models follow their respective licenses (Vosk, Hugging Face).
* Suitable for **viva demos**, **research presentations**, **performance comparison studies**, and **edge-vs-cloud analysis**.

---

## 🧩 Future Improvements (Recommended for Higher Marks)

You can mention these during viva to show research depth:

* Add Whisper-Tiny or Whisper-Base for additional comparison
* Add WER computation using reference test sets
* Add noise-robustness evaluation
* Add streaming ASR pipeline
* Add GPU acceleration
* Deploy using Docker or Render

---

## 📎 References

Curated links for academic context and implementation guidance (numbered as in your original list).

[1] PyTorch Audio ASR Tutorial
[2] Wav2Vec2 Model Card
[3–6] Vosk Documentation & Implementation Guides
[7–8] ASR Evaluation Repositories
[9–20] Practical ASR Projects, APIs & Tutorials




