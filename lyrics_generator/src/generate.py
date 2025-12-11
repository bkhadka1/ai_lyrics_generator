"""
Generate lyrics using fine-tuned GPT-2 model.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Optional


class LyricsGenerator:
    def __init__(self, model_path: str = "models/gpt2-lyrics"):
        """
        Initialize lyrics generator with trained model.
        
        Args:
            model_path: Path to fine-tuned model
        """
        print(f"Loading model from {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.2
    ) -> list:
        """
        Generate lyrics from prompt.
        
        Args:
            prompt: Starting text/theme for lyrics
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            num_return_sequences: Number of variations to generate
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            List of generated lyrics
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # Decode outputs
        generated_lyrics = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_lyrics.append(text)
        
        return generated_lyrics
    
    def format_lyrics(self, text: str) -> str:
        """
        Format generated text to look like proper lyrics.
        
        Args:
            text: Raw generated text
            
        Returns:
            Formatted lyrics
        """
        # Split into lines
        lines = text.split('\n')
        
        # Clean up lines
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # Capitalize first letter
                line = line[0].upper() + line[1:] if len(line) > 1 else line.upper()
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)


def main():
    """Demo generation."""
    generator = LyricsGenerator()
    
    # Example prompts
    prompts = [
        "Heartbreak in the rain",
        "Dancing under moonlight",
        "Lost in the city lights"
    ]
    
    print("\n=== Generating Lyrics ===\n")
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        print("-" * 50)
        
        lyrics = generator.generate(
            prompt=prompt,
            max_length=150,
            temperature=0.8,
            num_return_sequences=1
        )
        
        formatted = generator.format_lyrics(lyrics[0])
        print(formatted)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()