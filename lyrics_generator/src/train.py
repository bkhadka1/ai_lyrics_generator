"""
Fine-tune GPT-2 on lyrics dataset using Hugging Face Transformers.
Updated to work with Kaggle dataset loader.
"""

import os
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
import torch


class LyricsTrainer:
    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: str = "models/gpt2-lyrics",
        max_length: int = 512
    ):
        """
        Initialize trainer for lyrics generation.
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save model
            max_length: Maximum sequence length for training
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        
        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
    
    def prepare_dataset(self, csv_path: str, text_col: str = 'lyrics'):
        """
        Load and tokenize dataset.
        
        Args:
            csv_path: Path to CSV file (processed or cleaned)
            text_col: Column name containing lyrics
            
        Returns:
            Tokenized dataset
        """
        print(f"\nLoading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Ensure lyrics column exists
        if text_col not in df.columns:
            # Try alternative column names
            alternatives = ['cleaned_lyrics', 'lyric', 'text']
            for alt in alternatives:
                if alt in df.columns:
                    text_col = alt
                    print(f"Using column: {text_col}")
                    break
            else:
                raise ValueError(f"No lyrics column found. Available: {df.columns.tolist()}")
        
        # Remove nulls
        df = df[df[text_col].notna()].copy()
        
        print(f"Dataset size: {len(df):,} examples")
        print(f"Avg length: {df[text_col].str.len().mean():.0f} chars")
        
        # Create HuggingFace dataset
        dataset = Dataset.from_pandas(df[[text_col]])
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_col],
                truncation=True,
                max_length=self.max_length,
                padding=False
            )
        
        print("Tokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing lyrics"
        )
        
        # Group texts into blocks
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            
            # Drop remainder
            total_length = (total_length // self.max_length) * self.max_length
            
            result = {
                k: [t[i:i + self.max_length] for i in range(0, total_length, self.max_length)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        print("Grouping texts into blocks...")
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            desc="Grouping texts"
        )
        
        print(f"Final dataset: {len(lm_dataset):,} training blocks")
        
        return lm_dataset
    
    def train(
        self,
        train_dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        save_steps: int = 1000,
        eval_steps: int = None
    ):
        """
        Train the model.
        
        Args:
            train_dataset: Prepared training dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            warmup_steps: Warmup steps for scheduler
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps (None = no eval)
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=100,
            save_steps=save_steps,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="none",
            logging_dir="./logs",
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"Training blocks: {len(train_dataset):,}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size} (effective: {batch_size * 4})")
        print(f"Learning rate: {learning_rate}")
        print("="*60 + "\n")
        
        trainer.train()
        
        # Save final model
        print(f"\n💾 Saving model to {self.output_dir}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved to: {self.output_dir}")
        print(f"\nNext step: Run the Streamlit app")
        print(f"  streamlit run app/streamlit_app.py")


def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GPT-2 on lyrics")
    parser.add_argument(
        "--data",
        type=str,
        default="data/lyrics_processed.csv",
        help="Path to processed CSV file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("\n💡 Run this first:")
        print("  python src/load_data.py")
        return
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = LyricsTrainer(
        model_name="gpt2",
        output_dir="models/gpt2-lyrics",
        max_length=512
    )
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(args.data)
    
    # Train model
    trainer.train(
        train_dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()