🎤 AI Lyrics Generator (GPT-2 Fine-Tuned)

An end-to-end Natural Language Processing project that fine-tunes GPT-2 on song lyrics and deploys a Streamlit web app for generating lyrics based on user prompts.
Built completely from scratch using Python, Hugging Face Transformers, and PyTorch.

🚀 Features

Fine-tune GPT-2 on custom lyrics dataset

Clean + preprocess raw text into training-ready format

Train autoregressive language model for lyric generation

Generate lyrics via prompt (mood, genre, theme, first line)

Interactive Streamlit Web App

Professional folder structure for production ML

Easy deployment & GitHub-ready

📁 Project Structure
lyrics-generator/
│
├── data/
│   └── lyrics.csv
│
├── models/
│   └── gpt2-lyrics/
│
├── src/
│   ├── clean_data.py
│   ├── tokenize.py
│   ├── train.py
│   ├── generate.py
│   └── utils.py
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
└── .gitignore

🧠 Model Architecture

This project fine-tunes GPT-2 (124M parameters) using causal language modeling (CLM).
Training strategy includes:

Block-level grouping (512 tokens)

Mixed precision (FP16)

Gradient accumulation

Tokenizer extended with EOS as pad token

Temperature & top-p sampling for natural creativity

🔧 Setup Instructions
1. Clone the Repository
git clone https://github.com/<YOUR_USERNAME>/lyrics-generator.git
cd lyrics-generator

2. Install Dependencies
pip install -r requirements.txt

3. Add Your Dataset

Place lyrics.csv inside /data/.

Required columns:

lyrics
artist (optional)
title (optional)

4. Preprocess + Tokenize
python src/tokenize.py

5. Fine-Tune GPT-2
python src/train.py

6. Run the Streamlit App
streamlit run app/streamlit_app.py

🎧 Example Generation

Prompt:

"heartbreak in the rain"


Output:

Heartbreak in the rain,
I’m walking through the shadows of your name,
The memories fall like thunder in the dark,
Trying to find a fire from a spark...

🧪 Tech Stack
Component	Technology
Language Model	GPT-2
Framework	PyTorch
Tokenizer	GPT2TokenizerFast
Training	HuggingFace Trainer
Data	HuggingFace datasets
Web App	Streamlit
Deployment	Streamlit Cloud / Local
