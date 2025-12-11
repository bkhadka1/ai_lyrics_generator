"""
Data cleaning and preprocessing for lyrics dataset.
Handles text normalization, filtering, and formatting.
"""

import pandas as pd
import re
from typing import List, Optional
import unicodedata


class LyricsPreprocessor:
    def __init__(self, min_length: int = 50, max_length: int = 5000):
        """
        Initialize preprocessor with length constraints.
        
        Args:
            min_length: Minimum character count for valid lyrics
            max_length: Maximum character count for valid lyrics
        """
        self.min_length = min_length
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """Clean individual text entry."""
        if pd.isna(text):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\'\",\.\!\?\-\n]', '', text)
        
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def is_valid_lyrics(self, text: str) -> bool:
        """Check if lyrics meet quality criteria."""
        if not text or len(text) < self.min_length:
            return False
        
        if len(text) > self.max_length:
            return False
        
        # Check if it has enough actual words (not just spaces/newlines)
        words = text.split()
        if len(words) < 10:
            return False
        
        return True
    
    def process_dataset(self, df: pd.DataFrame, lyrics_col: str = 'lyrics') -> pd.DataFrame:
        """
        Process entire dataset.
        
        Args:
            df: Input dataframe with lyrics
            lyrics_col: Name of the column containing lyrics
            
        Returns:
            Cleaned dataframe
        """
        print(f"Original dataset size: {len(df)}")
        
        # Clean lyrics
        df['cleaned_lyrics'] = df[lyrics_col].apply(self.clean_text)
        
        # Filter valid entries
        df['is_valid'] = df['cleaned_lyrics'].apply(self.is_valid_lyrics)
        df_clean = df[df['is_valid']].copy()
        
        print(f"Cleaned dataset size: {len(df_clean)}")
        print(f"Removed {len(df) - len(df_clean)} invalid entries")
        
        return df_clean[['cleaned_lyrics'] + [col for col in df.columns if col not in ['cleaned_lyrics', 'is_valid']]]


def main():
    """Main execution function."""
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data/lyrics.csv')
    
    # Initialize preprocessor
    preprocessor = LyricsPreprocessor(min_length=50, max_length=5000)
    
    # Clean data
    print("\nCleaning lyrics...")
    df_clean = preprocessor.process_dataset(df)
    
    # Save cleaned data
    output_path = 'data/lyrics_cleaned.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to {output_path}")
    
    # Display statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total songs: {len(df_clean)}")
    print(f"Avg lyrics length: {df_clean['cleaned_lyrics'].str.len().mean():.0f} chars")
    print(f"Median lyrics length: {df_clean['cleaned_lyrics'].str.len().median():.0f} chars")


if __name__ == "__main__":
    main()