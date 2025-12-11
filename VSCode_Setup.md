# 🖥️ VS Code Local Setup Guide (After Cloning)

Complete step-by-step guide to run the AI Lyrics Generator locally after cloning from GitHub.

---

## ✅ Current Status
- ✅ Code cloned from GitHub
- ✅ Opening in VS Code
- 🎯 **Goal:** Get it running locally

---

## 🚀 Step-by-Step Instructions

### **STEP 1: Open Project in VS Code**

```bash
# Navigate to your cloned folder
cd path/to/lyrics-generator

# Open in VS Code
code .
```

Or: File → Open Folder → Select `lyrics-generator`

---

### **STEP 2: Verify Folder Structure**

Check that you have these files/folders:

```
lyrics-generator/
├── data/                    # Empty (we'll fill this)
├── models/                  # Empty (created during training)
├── src/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── train.py
│   └── generate.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

✅ **If you see these, you're good to go!**

---

### **STEP 3: Create Virtual Environment**

Open VS Code Terminal: **`Ctrl + `` (backtick)** or **View → Terminal**

```bash
# Create virtual environment
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

✅ **You should see `(venv)` at the start of your terminal line**

---

### **STEP 4: Install Dependencies**

```bash
# Make sure venv is activated (you see (venv) in terminal)
pip install -r requirements.txt
```

⏳ **This will take 2-5 minutes.** It's installing:
- PyTorch
- Transformers
- Streamlit
- Kagglehub
- Other dependencies

**Expected output:**
```
Collecting torch>=2.0.0
Collecting transformers>=4.30.0
...
Successfully installed torch-2.x.x transformers-4.x.x ...
```

---

### **STEP 5: Setup Kaggle API (One-Time Setup)**

#### 5a. Get Kaggle API Credentials

1. Go to [kaggle.com/account](https://www.kaggle.com/account)
2. Scroll down to **"API"** section
3. Click **"Create New API Token"**
4. This downloads `kaggle.json` to your Downloads folder

#### 5b. Install Kaggle Credentials

**Windows:**
```bash
# Create kaggle folder
mkdir %USERPROFILE%\.kaggle

# Move the file (update path if needed)
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

**Mac/Linux:**
```bash
# Create kaggle folder
mkdir -p ~/.kaggle

# Move the file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

#### 5c. Verify Setup

```bash
# This should NOT give an error
python -c "import kagglehub; print('Kaggle API ready!')"
```

✅ **Expected:** `Kaggle API ready!`

---

### **STEP 6: Accept Dataset Terms (IMPORTANT!)**

⚠️ **You MUST do this or the download will fail:**

1. Visit: https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information
2. Click the **"Download"** button (you don't actually download, just accept terms)
3. You should see a popup asking you to accept terms → Click **"I Understand and Accept"**

✅ **Done!** Now the script can download it programmatically.

---

### **STEP 7: Download and Process Dataset**

```bash
# Run the data loader script
python src/load_data.py
```

**You'll see this menu:**
```
🎤 AI LYRICS GENERATOR - DATA PREPARATION

Choose mode:
  1. Quick test (1,000 songs, ~5 min training)
  2. Small dataset (10,000 songs, ~30 min training)
  3. Full dataset (all songs, 2-4 hours training)

Enter choice (1/2/3):
```

**For first time, choose `1` (test mode)**

Type: `1` and press Enter

**Expected output:**
```
🎲 Test mode: Using 1,000 songs

📥 Downloading dataset from Kaggle...
(This may take a few minutes on first run)
✅ Dataset downloaded to: C:\Users\...\kagglehub\...

📖 Loading data from: song_lyrics.csv
✅ Loaded 1,000,000 total songs
🌍 Filtered to en: 650,000 songs
📏 Filtered by length (>=100 chars): 580,000 songs
🎲 Sampled for testing: 1,000 songs

💾 Saving processed data to: data\lyrics_processed.csv

📊 DATASET STATISTICS
====================================
Total songs: 1,000
Avg lyrics length: 834 chars
Median lyrics length: 712 chars
Unique artists: 876

✅ Dataset ready for training!
```

✅ **You should now have:** `data/lyrics_processed.csv`

⏳ **Time:** 30 seconds - 2 minutes (downloads once, then caches)

---

### **STEP 8: Train the Model**

```bash
# Start training
python src/train.py
```

**Expected output:**
```
Initializing trainer...
Loading gpt2...
Model loaded: 124.4M parameters

Loading data from data\lyrics_processed.csv...
Dataset size: 1,000 examples
Avg length: 834 chars
Tokenizing...
Grouping texts into blocks...
Final dataset: 1,623 training blocks

============================================================
🚀 STARTING TRAINING
============================================================
Device: cuda (or cpu)
Training blocks: 1,623
Epochs: 3
Batch size: 4 (effective: 16)
Learning rate: 5e-05
============================================================

[Progress bars showing training...]
Epoch 1/3: 100%|████████| 102/102 [01:23<00:00,  1.22it/s]
Epoch 2/3: 100%|████████| 102/102 [01:21<00:00,  1.25it/s]
Epoch 3/3: 100%|████████| 102/102 [01:20<00:00,  1.27it/s]

💾 Saving model to models/gpt2-lyrics...

============================================================
✅ TRAINING COMPLETE!
============================================================
Model saved to: models/gpt2-lyrics
```

⏳ **Expected Time:**
- **Test mode (1K songs):** 5-10 minutes (CPU) or 2-3 minutes (GPU)
- **Small mode (10K songs):** 30-45 minutes (CPU) or 10-15 minutes (GPU)
- **Full mode:** 3-6 hours (CPU) or 1-2 hours (GPU)

✅ **You should now have:** `models/gpt2-lyrics/` folder with trained model

---

### **STEP 9: Test Generation (Optional)**

```bash
# Test if the model works
python src/generate.py
```

**Expected output:**
```
Loading model from models/gpt2-lyrics...
Model loaded on cpu

=== Generating Lyrics ===

Prompt: 'Heartbreak in the rain'
--------------------------------------------------
Heartbreak in the rain
I'm walking through the shadows of your name
The memories fall like thunder in the dark
Trying to find a fire from a spark
...

==================================================
```

✅ **If you see generated lyrics, it's working!**

---

### **STEP 10: Launch Streamlit App**

```bash
# Start the web app
streamlit run app/streamlit_app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

✅ **Your browser should automatically open to the app!**

🎉 **You should see the AI Lyrics Generator interface!**

---

## 🎨 Using the Web App

1. **Enter a prompt** in the text box:
   - "Heartbreak in the rain"
   - "Summer love on the beach"
   - "Lost in the city lights"

2. **Adjust settings** in the sidebar:
   - Temperature (creativity)
   - Max length
   - Repetition penalty

3. **Click "🎵 Generate Lyrics"**

4. **View your generated lyrics!**

---

## 🔧 Troubleshooting

### ❌ Issue: "Module not found"
**Fix:**
```bash
# Make sure venv is activated (you see (venv))
# If not, reactivate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

### ❌ Issue: "kaggle.json not found"
**Fix:**
```bash
# Check if file exists:
# Windows:
dir %USERPROFILE%\.kaggle

# Mac/Linux:
ls ~/.kaggle

# If not there, repeat STEP 5
```

---

### ❌ Issue: "401 Unauthorized" when downloading
**Fix:**
1. Re-download `kaggle.json` from kaggle.com/account
2. Replace the old one in `.kaggle/` folder
3. Make sure you accepted dataset terms (STEP 6)

---

### ❌ Issue: Streamlit can't find module
**Fix:**
```bash
# Streamlit needs to know where to find src/
# Run from project root:
cd lyrics-generator
streamlit run app/streamlit_app.py
```

---

### ❌ Issue: Training is very slow
**Options:**
1. Use test mode (1K songs) for faster testing
2. Install GPU PyTorch if you have NVIDIA GPU:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
3. Reduce batch size in `train.py` (line 100): `batch_size=2`

---

### ❌ Issue: Out of memory during training
**Fix:**
```bash
# Edit src/train.py
# Find line ~100 and change:
batch_size=2  # Reduce from 4 to 2
```

---

## 📊 Performance Expectations

| Hardware | Training Time (1K songs) |
|----------|--------------------------|
| CPU (Intel i5/i7) | 8-12 minutes |
| CPU (M1/M2 Mac) | 5-8 minutes |
| GPU (GTX 1060) | 3-4 minutes |
| GPU (RTX 3060+) | 1-2 minutes |

---

## 🎯 Success Checklist

After completing all steps, you should have:

- ✅ Virtual environment activated `(venv)`
- ✅ All dependencies installed
- ✅ Kaggle API configured
- ✅ Dataset downloaded to `data/lyrics_processed.csv`
- ✅ Model trained in `models/gpt2-lyrics/`
- ✅ Streamlit app running at `localhost:8501`
- ✅ Able to generate lyrics!

---

## 🚀 What's Next?

### Train on More Data
```bash
# Run load_data.py again and choose option 2 or 3
python src/load_data.py
# Then retrain
python src/train.py
```

### Customize Generation
Edit `app/streamlit_app.py` to:
- Add genre selection
- Add artist style mimicking
- Save generated lyrics to file

### Deploy Online
- Push to GitHub
- Deploy on Streamlit Cloud (free)
- Share your app with the world!

---

## 💡 Quick Reference Commands

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Download data
python src/load_data.py

# Train model
python src/train.py

# Test generation
python src/generate.py

# Launch app
streamlit run app/streamlit_app.py

# Deactivate environment (when done)
deactivate
```

---

## 📧 Need Help?

If you're stuck at any step:
1. Check the error message carefully
2. Refer to the Troubleshooting section above
3. Make sure your virtual environment is activated
4. Verify all files are in the correct folders

---

**🎉 You're all set! Enjoy creating AI-generated lyrics!**