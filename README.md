# 🎤 AI Lyrics Generator

An end-to-end Natural Language Processing project that fine-tunes GPT-2 on song lyrics and deploys a Streamlit web app for generating lyrics based on user prompts. Built completely from scratch using Python, Hugging Face Transformers, and PyTorch.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)

---

## 🚀 Features

- ✅ **Fine-tune GPT-2** on custom lyrics dataset
- ✅ **Clean & preprocess** raw text into training-ready format
- ✅ **Train autoregressive language model** for lyric generation
- ✅ **Generate lyrics** via prompt (mood, genre, theme, first line)
- ✅ **Interactive Streamlit Web App** with real-time generation
- ✅ **Professional folder structure** for production ML
- ✅ **Easy deployment** & GitHub-ready

---

## 📁 Project Structure

```
lyrics-generator/
│
├── data/
│   ├── lyrics.csv              # Raw lyrics dataset
│   └── lyrics_cleaned.csv      # Preprocessed data
│
├── models/
│   └── gpt2-lyrics/            # Fine-tuned model checkpoint
│
├── src/
│   ├── clean_data.py           # Data cleaning pipeline
│   ├── train.py                # Model training script
│   ├── generate.py             # Lyrics generation module
│   └── utils.py                # Helper functions
│
├── app/
│   └── streamlit_app.py        # Web interface
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

---

## 🧠 Model Architecture

This project fine-tunes **GPT-2 (124M parameters)** using **causal language modeling (CLM)**. 

### Training Strategy:
- **Block-level grouping** (512 tokens)
- **Mixed precision** (FP16) for faster training
- **Gradient accumulation** (4 steps)
- **Tokenizer extended** with EOS as pad token
- **Temperature & top-p sampling** for natural creativity

---

## 🔧 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/lyrics-generator.git
cd lyrics-generator
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**For GPU support (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4️⃣ Add Your Dataset

Place `lyrics.csv` inside `/data/`.

**Required columns:**
```
lyrics          # (required) - Song lyrics text
artist          # (optional) - Artist name
title           # (optional) - Song title
```

**Example format:**
```csv
lyrics,artist,title
"Verse 1: ...
Chorus: ...
Verse 2: ...",Artist Name,Song Title
```

---

## 🎯 Usage Guide

### Step 1: Clean & Preprocess Data

```bash
python src/clean_data.py
```

**What it does:**
- Removes special characters
- Normalizes unicode
- Filters songs by length
- Saves cleaned data to `data/lyrics_cleaned.csv`

---

### Step 2: Fine-Tune GPT-2

```bash
python src/train.py
```

**Training parameters:**
- **Epochs:** 3
- **Batch size:** 4 (with gradient accumulation)
- **Learning rate:** 5e-5
- **Max sequence length:** 512 tokens

**Expected training time:**
- **CPU:** 12-24 hours (for ~10k songs)
- **GPU (T4/V100):** 2-4 hours

Model will be saved to `models/gpt2-lyrics/`

---

### Step 3: Test Generation (Optional)

```bash
python src/generate.py
```

Generates sample lyrics from predefined prompts to verify model quality.

---

### Step 4: Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Access at: `http://localhost:8501`

---

## 🎧 Example Generation

### Prompt:
```
"heartbreak in the rain"
```

### Output:
```
Heartbreak in the rain,
I'm walking through the shadows of your name,
The memories fall like thunder in the dark,
Trying to find a fire from a spark.

Every drop reminds me of your touch,
Now the silence says too much,
I'm drowning in the echo of goodbye,
Watching all our moments pass me by.
```

---

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Free)

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Deploy `app/streamlit_app.py`

**Note:** Upload your trained model to the repo or use Hugging Face model hub.

### Option 2: Hugging Face Spaces

```bash
# Push model to Hugging Face
huggingface-cli login
huggingface-cli repo create lyrics-generator
git push huggingface main
```

### Option 3: Local/Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

---

## 🧪 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language Model** | GPT-2 (124M) |
| **Framework** | PyTorch 2.0+ |
| **Tokenizer** | GPT2TokenizerFast |
| **Training** | Hugging Face Trainer |
| **Data** | Hugging Face Datasets |
| **Web App** | Streamlit |
| **Deployment** | Streamlit Cloud / Hugging Face |

---

## 📊 Dataset Requirements

### Recommended Size:
- **Minimum:** 1,000 songs
- **Optimal:** 10,000+ songs
- **Professional:** 50,000+ songs

### Data Sources:
- [Kaggle Lyrics Datasets](https://www.kaggle.com/datasets?search=lyrics)
- [Genius API](https://genius.com/api-clients)
- [AZLyrics scraper](https://www.azlyrics.com/) (follow robots.txt)

---

## 🎛️ Hyperparameter Tuning

Adjust in `src/train.py`:

```python
# Creativity vs Coherence
temperature = 0.8          # 0.7-1.0 recommended
top_p = 0.9                # Nucleus sampling
repetition_penalty = 1.2   # Avoid repetition

# Training
learning_rate = 5e-5       # 3e-5 to 1e-4
num_epochs = 3             # 2-5 epochs
batch_size = 4             # Depends on GPU
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)
**Solution:**
- Reduce `batch_size` in `train.py`
- Reduce `max_length` to 256 or 384
- Enable gradient checkpointing

### Issue: Repetitive Output
**Solution:**
- Increase `repetition_penalty` (1.3-1.5)
- Adjust `no_repeat_ngram_size=3`
- Lower `temperature`

### Issue: Model Not Found
**Solution:**
```bash
# Check if model exists
ls models/gpt2-lyrics/

# If not, retrain:
python src/train.py
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Commit changes
4. Open a pull request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [OpenAI GPT-2](https://openai.com/research/gpt-2)
- [Streamlit](https://streamlit.io/)

---

## 📧 Contact

**Bikash Khadka**  
GitHub: [@bkhadka1](https://github.com/bkhadka1)  
Email: bcash2233@gmail.com

---

**⭐ If this project helped you, give it a star!**
