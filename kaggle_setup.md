# 📦 Kaggle Dataset Setup Guide

This guide shows you how to use the large Kaggle dataset without adding it to GitHub.

---

## 🎯 The Problem

The Kaggle dataset is **too large for GitHub** (100MB+ limit). We need to:
- ✅ Download it locally
- ✅ Exclude it from git
- ✅ Make it easy for others to reproduce

---

## 🔧 Solution: Automated Download Script

We've created `load_data.py` that automatically downloads and processes the dataset.

---

## 📝 Step-by-Step Setup

### 1️⃣ Install Kaggle API

```bash
pip install kagglehub
```

### 2️⃣ Setup Kaggle Credentials

**Option A: Kaggle API Token (Recommended)**

1. Go to [kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`
5. Move it to the right location:

```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

**Option B: Environment Variables**

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 3️⃣ Accept Dataset Terms

⚠️ **IMPORTANT:** You must accept the dataset terms on Kaggle before downloading.

1. Visit: https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information
2. Click "Download" (you don't need to actually download)
3. Accept the terms if prompted

### 4️⃣ Download and Process Dataset

**Interactive Mode (Recommended for first time):**

```bash
python src/load_data.py
```

This will:
- Download the dataset (cached automatically by kagglehub)
- Ask you to choose: Test (1K songs), Small (10K), or Full
- Filter to English lyrics
- Remove short/invalid entries
- Save to `data/lyrics_processed.csv`

**Example Output:**
```
🎤 AI LYRICS GENERATOR - DATA PREPARATION

Choose mode:
  1. Quick test (1,000 songs, ~5 min training)
  2. Small dataset (10,000 songs, ~30 min training)
  3. Full dataset (all songs, 2-4 hours training)

Enter choice (1/2/3): 1

📥 Downloading dataset from Kaggle...
✅ Dataset downloaded to: /home/user/.cache/kagglehub/...
📖 Loading data from: song_lyrics.csv
✅ Loaded 1,000,000 total songs
🌍 Filtered to en: 650,000 songs
📏 Filtered by length (>=100 chars): 580,000 songs
🎲 Sampled for testing: 1,000 songs

💾 Saving processed data to: data/lyrics_processed.csv
✅ Dataset ready for training!
```

---

## 🗂️ Project Structure After Setup

```
lyrics-generator/
│
├── data/
│   └── lyrics_processed.csv        # ✅ Generated (gitignored)
│
├── models/
│   └── gpt2-lyrics/                # ✅ Generated after training (gitignored)
│
├── src/
│   ├── load_data.py                # ✅ New: Downloads Kaggle data
│   ├── clean_data.py               # Optional: Extra cleaning
│   ├── train.py                    # ✅ Updated: Uses processed data
│   ├── generate.py
│   └── utils.py
│
├── .gitignore                      # ✅ Updated: Excludes data/models
└── requirements.txt                # ✅ Updated: Includes kagglehub
```

---

## 🚀 Complete Workflow

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Download and process data (ONE TIME)
python src/load_data.py

# 3. Train model
python src/train.py

# 4. Launch app
streamlit run app/streamlit_app.py
```

---

## 🔄 For Other Users / Collaborators

When someone clones your repo, they just run:

```bash
git clone https://github.com/yourusername/lyrics-generator.git
cd lyrics-generator
pip install -r requirements.txt
python src/load_data.py  # Downloads data automatically
python src/train.py
```

**No manual dataset download needed!** 🎉

---

## 📊 Dataset Sizes & Training Times

| Mode | Songs | Download | Train (GPU) | Train (CPU) |
|------|-------|----------|-------------|-------------|
| **Test** | 1,000 | 30s | 5 min | 30 min |
| **Small** | 10,000 | 30s | 30 min | 3 hours |
| **Full** | 500,000+ | 30s | 2-4 hours | 24+ hours |

---

## 🎛️ Advanced: Custom Filtering

Edit `src/load_data.py` to customize:

```python
df = loader.load_and_filter(
    language="en",           # Language code
    min_length=200,          # Longer lyrics only
    max_songs=50000,         # Custom limit
    sample_for_testing=False # Full dataset
)
```

---

## 🐛 Troubleshooting

### Issue: "401 Unauthorized"
**Solution:**
- Make sure `~/.kaggle/kaggle.json` exists
- Verify permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Re-download API token from Kaggle

### Issue: "403 Forbidden"
**Solution:**
- Accept dataset terms on Kaggle website
- Visit dataset page and click "Download"

### Issue: Dataset already downloaded but script re-downloads
**Solution:**
- Kagglehub caches downloads automatically
- Check: `~/.cache/kagglehub/` (Linux/Mac) or `%LOCALAPPDATA%\kagglehub\` (Windows)
- Script will use cached version if available

### Issue: Out of disk space
**Solution:**
- Test mode uses only 1,000 songs (~1MB)
- Delete cache: `rm -rf ~/.cache/kagglehub/`
- Use smaller `max_songs` parameter

---

## 📦 What Gets Added to Git

✅ **Include:**
- All Python scripts (`src/*.py`, `app/*.py`)
- Configuration files (`requirements.txt`, `.gitignore`)
- Documentation (`README.md`, `KAGGLE_SETUP.md`)

❌ **Exclude (.gitignore):**
- `data/*.csv` - Dataset files
- `models/` - Trained models
- `__pycache__/` - Python cache
- `.kaggle/` - Kaggle credentials

---

## 🌐 Deployment Options

### Option 1: Streamlit Cloud
1. Train model locally
2. Upload model to Hugging Face Hub:
```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("models/gpt2-lyrics")
model.push_to_hub("your-username/gpt2-lyrics")
```
3. Update `generate.py` to load from Hub
4. Deploy to Streamlit Cloud (no dataset needed)

### Option 2: Hugging Face Spaces
```bash
# Upload model
huggingface-cli login
huggingface-cli upload your-username/gpt2-lyrics models/gpt2-lyrics

# Deploy Space with model
```

### Option 3: Local/Server
- Train and run on same machine
- Dataset stays local
- Full control

---

## 💡 Pro Tips

1. **Start with test mode** - Verify pipeline works before full training
2. **Use GPU** - 10-20x faster than CPU
3. **Cache models** - Keep trained models, re-download data only if needed
4. **Version models** - Save checkpoints with dates: `gpt2-lyrics-2024-12-11`
5. **Monitor disk** - Full dataset + models can use 5-10GB

---

## 📚 References

- **Kaggle API Docs**: https://github.com/Kaggle/kaggle-api
- **Dataset**: https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information
- **Hugging Face Hub**: https://huggingface.co/docs/hub/index

---

**Questions?** Open an issue or check the main README.md
