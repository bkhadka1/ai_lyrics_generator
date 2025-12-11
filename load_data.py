"""
Download and load Kaggle dataset for lyrics generation.
Handles automatic download and preprocessing.
"""

import os
import pandas as pd
import kagglehub
from pathlib import Path


class DatasetLoader:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory to store processed data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.raw_path = None
        self.processed_path = self.data_dir / "lyrics_processed.csv"
    
    def download_from_kaggle(self, force_download: bool = False):
        """
        Download dataset from Kaggle using kagglehub.
        
        Args:
            force_download: Re-download even if already cached
            
        Returns:
            Path to downloaded dataset
        """
        print("📥 Downloading dataset from Kaggle...")
        print("(This may take a few minutes on first run)")
        
        try:
            # Download using kagglehub (caches automatically)
            path = kagglehub.dataset_download(
                "carlosgdcj/genius-song-lyrics-with-language-information"
            )
            
            self.raw_path = Path(path)
            print(f"✅ Dataset downloaded to: {self.raw_path}")
            
            # List available files
            files = list(self.raw_path.glob("*.csv"))
            print(f"\n📁 Available files: {[f.name for f in files]}")
            
            return self.raw_path
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("\n💡 Make sure you have:")
            print("   1. Kaggle API credentials (~/.kaggle/kaggle.json)")
            print("   2. Accepted dataset terms on Kaggle website")
            raise
    
    def load_and_filter(
        self, 
        language: str = "en",
        min_length: int = 100,
        max_songs: int = None,
        sample_for_testing: bool = False
    ):
        """
        Load and filter the dataset.
        
        Args:
            language: Filter by language code (e.g., 'en' for English)
            min_length: Minimum lyrics length in characters
            max_songs: Maximum number of songs to load (None = all)
            sample_for_testing: If True, loads only 1000 songs for quick testing
            
        Returns:
            Filtered DataFrame
        """
        # Download if not already done
        if self.raw_path is None:
            self.download_from_kaggle()
        
        # Find the main CSV file
        csv_files = list(self.raw_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_path}")
        
        # Use the largest CSV (usually the main dataset)
        main_csv = max(csv_files, key=lambda f: f.stat().st_size)
        print(f"\n📖 Loading data from: {main_csv.name}")
        
        # Load with progress
        df = pd.read_csv(main_csv)
        print(f"✅ Loaded {len(df):,} total songs")
        
        # Filter by language if column exists
        if 'language' in df.columns:
            original_len = len(df)
            df = df[df['language'] == language].copy()
            print(f"🌍 Filtered to {language}: {len(df):,} songs ({len(df)/original_len*100:.1f}%)")
        
        # Filter by lyrics length
        if 'lyrics' in df.columns:
            df['lyrics_length'] = df['lyrics'].str.len()
            df = df[df['lyrics_length'] >= min_length].copy()
            print(f"📏 Filtered by length (>={min_length} chars): {len(df):,} songs")
        
        # Remove null lyrics
        df = df[df['lyrics'].notna()].copy()
        
        # Sample for testing
        if sample_for_testing:
            df = df.sample(n=min(1000, len(df)), random_state=42)
            print(f"🎲 Sampled for testing: {len(df):,} songs")
        
        # Limit max songs
        if max_songs and len(df) > max_songs:
            df = df.sample(n=max_songs, random_state=42)
            print(f"🔢 Limited to max songs: {len(df):,} songs")
        
        return df
    
    def prepare_for_training(
        self,
        language: str = "en",
        min_length: int = 100,
        max_songs: int = None,
        test_mode: bool = False
    ):
        """
        Complete pipeline: download, filter, and save processed data.
        
        Args:
            language: Language to filter
            min_length: Minimum lyrics length
            max_songs: Maximum songs to use
            test_mode: If True, uses only 1000 songs for quick testing
            
        Returns:
            Path to processed CSV
        """
        print("\n" + "="*60)
        print("🚀 LYRICS DATASET PREPARATION PIPELINE")
        print("="*60 + "\n")
        
        # Check if already processed
        if self.processed_path.exists() and not test_mode:
            print(f"✅ Found existing processed data at: {self.processed_path}")
            response = input("Use existing data? (y/n): ").lower()
            if response == 'y':
                return self.processed_path
        
        # Download and filter
        df = self.load_and_filter(
            language=language,
            min_length=min_length,
            max_songs=max_songs,
            sample_for_testing=test_mode
        )
        
        # Save processed data
        print(f"\n💾 Saving processed data to: {self.processed_path}")
        df.to_csv(self.processed_path, index=False)
        
        # Print statistics
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        print(f"Total songs: {len(df):,}")
        print(f"Avg lyrics length: {df['lyrics'].str.len().mean():.0f} chars")
        print(f"Median lyrics length: {df['lyrics'].str.len().median():.0f} chars")
        
        if 'artist' in df.columns:
            print(f"Unique artists: {df['artist'].nunique():,}")
        
        # Show sample
        print("\n📝 Sample lyrics (first 200 chars):")
        print("-" * 60)
        print(df['lyrics'].iloc[0][:200] + "...")
        
        print("\n✅ Dataset ready for training!")
        print(f"📂 Processed file: {self.processed_path}")
        
        return self.processed_path


def main():
    """Main execution with user-friendly prompts."""
    print("\n🎤 AI LYRICS GENERATOR - DATA PREPARATION\n")
    
    # Ask user for mode
    print("Choose mode:")
    print("  1. Quick test (1,000 songs, ~5 min training)")
    print("  2. Small dataset (10,000 songs, ~30 min training)")
    print("  3. Full dataset (all songs, 2-4 hours training)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    # Configure based on choice
    if choice == "1":
        test_mode = True
        max_songs = 1000
        print("\n🎲 Test mode: Using 1,000 songs")
    elif choice == "2":
        test_mode = False
        max_songs = 10000
        print("\n📦 Small dataset: Using 10,000 songs")
    else:
        test_mode = False
        max_songs = None
        print("\n🚀 Full dataset: Using all available songs")
    
    # Initialize loader
    loader = DatasetLoader()
    
    # Prepare dataset
    processed_path = loader.prepare_for_training(
        language="en",
        min_length=100,
        max_songs=max_songs,
        test_mode=test_mode
    )
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Run: python src/clean_data.py (optional, for extra cleaning)")
    print(f"  2. Run: python src/train.py (starts training)")
    print(f"  3. Run: streamlit run app/streamlit_app.py (after training)")
    

if __name__ == "__main__":
    main()