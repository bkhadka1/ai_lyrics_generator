"""
Streamlit web app for AI Lyrics Generator.
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate import LyricsGenerator


# Page configuration
st.set_page_config(
    page_title="AI Lyrics Generator",
    page_icon="🎤",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load model (cached for performance)."""
    return LyricsGenerator(model_path="models/gpt2-lyrics")


def main():
    # Header
    st.title("🎤 AI Lyrics Generator")
    st.markdown("### Generate creative song lyrics using fine-tuned GPT-2")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        max_length = st.slider(
            "📏 Max Length",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
            help="Maximum number of tokens to generate"
        )
        
        top_p = st.slider(
            "🎯 Top-p (Nucleus Sampling)",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Probability threshold for token selection"
        )
        
        repetition_penalty = st.slider(
            "🔁 Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=1.2,
            step=0.1,
            help="Penalty for repeating words/phrases"
        )
        
        num_variants = st.number_input(
            "🎲 Number of Variants",
            min_value=1,
            max_value=5,
            value=1,
            help="Generate multiple variations"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Prompt Tips")
        st.markdown("""
        - **Mood**: "Heartbreak in the rain"
        - **Theme**: "Summer love on the beach"
        - **Genre**: "Rock anthem about freedom"
        - **First Line**: "Walking down this empty road"
        - **Specific**: "Lost in the city lights at midnight"
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✏️ Your Prompt")
        prompt = st.text_area(
            "Enter your prompt or theme:",
            height=150,
            placeholder="e.g., 'Heartbreak in the rain' or 'Dancing under moonlight'",
            help="Describe the mood, theme, or start of your lyrics"
        )
        
        # Example prompts
        st.markdown("**Quick Examples:**")
        example_cols = st.columns(3)
        
        examples = [
            "Heartbreak in the rain",
            "Dancing under moonlight", 
            "Lost in the city lights"
        ]
        
        for i, example in enumerate(examples):
            if example_cols[i].button(example, key=f"ex_{i}"):
                prompt = example
                st.rerun()
        
        generate_button = st.button("🎵 Generate Lyrics", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("🎼 Generated Lyrics")
        
        if generate_button:
            if not prompt.strip():
                st.warning("⚠️ Please enter a prompt first!")
            else:
                with st.spinner("🎵 Generating lyrics..."):
                    try:
                        # Load model
                        generator = load_model()
                        
                        # Generate
                        lyrics_list = generator.generate(
                            prompt=prompt,
                            max_length=max_length,
                            temperature=temperature,
                            top_p=top_p,
                            num_return_sequences=num_variants,
                            repetition_penalty=repetition_penalty
                        )
                        
                        # Display results
                        for i, lyrics in enumerate(lyrics_list):
                            if num_variants > 1:
                                st.markdown(f"**Variant {i+1}:**")
                            
                            formatted = generator.format_lyrics(lyrics)
                            st.text_area(
                                f"Output {i+1}",
                                value=formatted,
                                height=300,
                                key=f"output_{i}",
                                label_visibility="collapsed"
                            )
                            
                            # Copy button
                            if st.button(f"📋 Copy to Clipboard", key=f"copy_{i}"):
                                st.success("Copied! (Use Ctrl+C)")
                            
                            if i < len(lyrics_list) - 1:
                                st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"❌ Error generating lyrics: {str(e)}")
                        st.info("💡 Make sure the model is trained and saved in 'models/gpt2-lyrics/'")
        else:
            st.info("👈 Configure settings and enter a prompt to generate lyrics!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with GPT-2, Hugging Face Transformers & Streamlit</p>
        <p>Fine-tuned on song lyrics dataset | Model: 124M parameters</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()