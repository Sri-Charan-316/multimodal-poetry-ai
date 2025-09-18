import streamlit as st
from PIL import Image
import random
import os
import tempfile

# Try to import gTTS
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Configure the page
st.set_page_config(
    page_title="🎭 Multimodal Poetry AI Generator",
    page_icon="🎭",
    layout="wide"
)

def translate_text(text, target_language):
    """Translate text to target language using deep-translator"""
    try:
        from deep_translator import GoogleTranslator
        
        lang_codes = {
            "English": "en",
            "Spanish": "es", 
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Japanese": "ja",
            "Chinese": "zh-CN",
            "Arabic": "ar",
            "Hindi": "hi"
        }
        
        if target_language != "English" and target_language in lang_codes:
            translator = GoogleTranslator(source='en', target=lang_codes[target_language])
            translated = translator.translate(text)
            return translated
        return text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def create_audio(text, language="en"):
    """Create audio from text using gTTS"""
    if not TTS_AVAILABLE:
        return None
    
    try:
        # Map language names to gTTS language codes
        lang_mapping = {
            "English": "en", "Spanish": "es", "French": "fr", 
            "German": "de", "Italian": "it", "Portuguese": "pt",
            "Russian": "ru", "Japanese": "ja", "Chinese": "zh",
            "Arabic": "ar", "Hindi": "hi"
        }
        
        lang_code = lang_mapping.get(language, "en")
        clean_text = text.replace("*", "").replace("#", "").replace("**", "")
        
        if not clean_text.strip():
            return None
        
        tts = gTTS(text=clean_text, lang=lang_code)
        temp_file = tempfile.mktemp(suffix='.mp3')
        tts.save(temp_file)
        
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            return temp_file
        return None
        
    except Exception as e:
        st.error(f"Audio generation error: {str(e)}")
        return None

def generate_poetry_with_ai(prompt, style="free verse", theme="nature"):
    """Generate enhanced poetry using sophisticated templates"""
    
    poetry_templates = {
        "nature": [
            f"""Beneath the vast expanse of {prompt} sky,
Where whispers of ancient winds carry dreams,
Nature's eternal symphony unfolds with grace,
In harmonious dance with all that breathes and lives.

The golden light cascades through emerald leaves,
Each dewdrop holds a universe of wonder,
Mountains stand as sentinels of time itself,
While rivers sing their timeless songs of peace.""",

            f"""In the garden where {prompt} dwells,
Morning light paints masterpieces on petals,
Each flower a verse in nature's living poem,
Dancing to rhythms older than memory.

The soil holds secrets of countless springs,
Roots reaching deep for wisdom underground,
While branches stretch toward dreams untold,
Creating sanctuary for wandering souls."""
        ],
        
        "love": [
            f"""Your {prompt} heart beats like celestial music,
In the symphony of my deepest dreams,
Love's eternal flame burns ever bright,
Uniting two souls in sacred dance.

Through seasons of joy and gentle sorrow,
Our hearts have learned to speak as one,
Creating poetry from simple moments,
Writing verses with our intertwined lives.""",

            f"""In moments shared with {prompt} beside me,
Time suspends its relentless journey,
Two hearts discover perfect synchrony,
In love's most sacred, tender embrace.

Your laughter becomes my morning music,
Your dreams interweave with my own,
Together we paint new constellations,
Across the canvas of our shared sky."""
        ],
        
        "adventure": [
            f"""Beyond the distant {prompt} horizon,
Where adventure calls with siren song,
Brave hearts answer without hesitation,
Ready to embrace the great unknown.

The path ahead winds through mysteries,
Each step a chance to discover truth,
Courage lights the darkest valleys,
Hope guides through the steepest climbs."""
        ],
        
        "dreams": [
            f"""In dreams of {prompt} and shimmering magic,
Imagination spreads its golden wings,
Soaring beyond the bounds of possible,
Into realms where wonder never sleeps.

Here gravity becomes a suggestion,
Time flows like honey mixed with starlight,
Colors exist that have no earthly names,
And music grows from seeds of pure thought."""
        ],
        
        "mystery": [
            f"""In shadows deep where {prompt} whispers,
Ancient secrets hold their breath,
Mystery wraps around each moment,
Like silk veils hiding sacred truth.

What lies beyond the edge of knowing,
Calls to hearts that dare to wonder,
Through corridors of time and space,
Seeking answers to life's riddles."""
        ]
    }
    
    # Haiku format
    if style == "haiku":
        haiku_templates = [
            f"""{prompt} blooms bright
In the garden of my heart—
Beauty never fades""",
            f"""Whispers of {prompt}
Carry dreams across the sky—
Peace flows like water""",
            f"""Morning light reveals
{prompt} in all its glory—
Nature's gift to us"""
        ]
        return random.choice(haiku_templates)
    
    # Select template from theme for free verse
    if theme in poetry_templates:
        template = random.choice(poetry_templates[theme])
        return template
    else:
        return f"""Inspiration flows like a mighty river,
{prompt} guides my trembling, eager pen,
Creating symphonies from simple words,
In verses born of heart and soul united.

Where dreams and reality gently blend,
Magic lives in every chosen phrase,
Each line a bridge across the void,
Connecting hearts through time and space."""

def main():
    st.title("🎭 Multimodal Cross-Language Poetry AI")
    st.markdown("*Generate beautiful poetry across multiple languages with AI*")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Language selection
        target_language = st.selectbox(
            "🌍 Target Language:",
            ["English", "Spanish", "French", "German", "Italian", 
             "Portuguese", "Russian", "Japanese", "Chinese", "Arabic", "Hindi"]
        )
        
        # Poetry style
        poetry_style = st.selectbox(
            "📝 Poetry Style:",
            ["free verse", "haiku"]
        )
        
        # Theme selection
        theme = st.selectbox(
            "🎨 Theme:",
            ["nature", "love", "adventure", "dreams", "mystery", "custom"]
        )
        
        # Mood
        mood = st.selectbox(
            "😊 Mood:",
            ["peaceful", "energetic", "romantic", "mysterious", "joyful", "melancholic"]
        )
        
        st.divider()
        
        # Voice-over settings
        st.header("🎵 Voice-Over")
        
        enable_voice = st.checkbox(
            "🔊 Enable Voice-Over",
            value=TTS_AVAILABLE,
            disabled=not TTS_AVAILABLE,
            help="Generate audio narration of your poem"
        )
        
        if not TTS_AVAILABLE:
            st.error("Install gTTS for voice features: pip install gTTS")
        
        # Show language info
        if target_language != "English":
            st.info(f"🔄 Poetry will be translated to {target_language}")
            
        if enable_voice and TTS_AVAILABLE:
            st.info(f"🎵 Voice-over will be in {target_language}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Text input
        if theme == "custom":
            prompt = st.text_area(
                "Enter your poetry inspiration:",
                placeholder="Write about anything that inspires you...",
                height=100
            )
        else:
            prompt = st.text_area(
                f"Describe your {theme} inspiration:",
                placeholder=f"Tell me about {theme} that moves you...",
                height=100
            )
        
        # Image upload
        st.subheader("🖼️ Visual Inspiration (Optional)")
        uploaded_file = st.file_uploader(
            "Upload an image for inspiration:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to inspire your poetry"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your inspiration image", use_column_width=True)
            
            # Add image inspiration to prompt
            image_descriptions = [
                "a scene of breathtaking natural beauty",
                "colors that dance with light and shadow", 
                "a moment frozen in pure magic",
                "an image that speaks directly to the soul",
                "beauty that transcends words"
            ]
            image_desc = random.choice(image_descriptions)
            if prompt.strip():
                prompt += f" inspired by {image_desc}"
            else:
                prompt = f"An image showing {image_desc}"
        
        # Generate button
        generate_btn = st.button("🎪 Generate Cross-Language Poetry", type="primary", use_container_width=True)
    
    with col2:
        st.header("✨ Generated Poetry")
        
        if generate_btn and prompt.strip():
            with st.spinner(f"🎭 Crafting your {target_language} poetry..."):
                # Generate poetry in English first
                english_poetry = generate_poetry_with_ai(prompt, poetry_style, theme)
                
                # Translate if needed
                if target_language != "English":
                    with st.spinner(f"🔄 Translating to {target_language}..."):
                        final_poetry = translate_text(english_poetry, target_language)
                else:
                    final_poetry = english_poetry
                
                # Generate audio if enabled
                audio_file = None
                if enable_voice and TTS_AVAILABLE:
                    with st.spinner(f"🎵 Creating voice-over in {target_language}..."):
                        audio_file = create_audio(final_poetry, target_language)
                
                # Display result
                st.markdown(f"### Your {target_language} Poem:")
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 25px;
                    border-radius: 15px;
                    color: white;
                    font-family: 'Georgia', serif;
                    line-height: 1.8;
                    margin: 15px 0;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                    border: 2px solid rgba(255, 255, 255, 0.1);
                ">
                <div style="text-align: center; font-size: 1.2em; font-weight: 300;">
                {final_poetry.replace(chr(10), '<br>')}
                </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Audio player
                if audio_file and enable_voice:
                    st.markdown("### 🎵 Listen to Your Poem:")
                    
                    try:
                        with open(audio_file, 'rb') as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format='audio/mp3')
                        st.success("🎵 Audio generated successfully!")
                    except Exception as e:
                        st.error(f"Audio playback error: {str(e)}")
                
                # Poem metadata
                metadata_text = f"**Language:** {target_language} | **Style:** {poetry_style} | **Theme:** {theme} | **Mood:** {mood}"
                if enable_voice and TTS_AVAILABLE and audio_file:
                    metadata_text += f" | **Voice-over:** ✅ Enabled"
                st.markdown(metadata_text)
                
                # Show original if translated
                if target_language != "English":
                    with st.expander("📖 View Original English Version"):
                        st.text(english_poetry)
                
                # Action buttons
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if st.button("🔄 Generate Another"):
                        st.rerun()
                
                with col_b:
                    # Download poem option
                    st.download_button(
                        "💾 Download Poem",
                        data=final_poetry,
                        file_name=f"poem_{target_language.lower()}_{theme}.txt",
                        mime="text/plain"
                    )
                
                with col_c:
                    # Download audio option
                    if audio_file and enable_voice and os.path.exists(audio_file):
                        with open(audio_file, 'rb') as f:
                            audio_bytes = f.read()
                        st.download_button(
                            "🎵 Download Audio",
                            data=audio_bytes,
                            file_name=f"poem_audio_{target_language.lower()}_{theme}.mp3",
                            mime="audio/mp3"
                        )
        
        elif generate_btn and not prompt.strip():
            st.warning("⚠️ Please enter some inspiration text or upload an image!")
    
    # Language showcase
    st.markdown("---")
    st.subheader("🌍 Supported Languages")
    
    languages_display = {
        "English": "🇺🇸", "Spanish": "🇪🇸", "French": "🇫🇷", 
        "German": "🇩🇪", "Italian": "🇮🇹", "Portuguese": "🇵🇹",
        "Russian": "🇷🇺", "Japanese": "🇯🇵", "Chinese": "🇨🇳", 
        "Arabic": "🇸🇦", "Hindi": "🇮🇳"
    }
    
    cols = st.columns(6)
    for i, (lang, flag) in enumerate(languages_display.items()):
        with cols[i % 6]:
            st.markdown(f"{flag} {lang}")
    
    # Example section
    with st.expander("💡 Cross-Language Poetry Examples"):
        st.markdown("""
        **English:** *"Beneath the vast expanse of sunset sky, where whispers of ancient winds carry dreams..."*
        
        **Spanish:** *"Bajo la vasta extensión del cielo del atardecer, donde los susurros de vientos antiguos llevan sueños..."*
        
        **French:** *"Sous la vaste étendue du ciel du coucher du soleil, où les murmures des vents anciens portent des rêves..."*
        
        **Japanese:** *"夕日の空の広大な広がりの下で、古代の風のささやきが夢を運ぶところ..."*
        
        Try different themes and languages to explore poetry across cultures!
        
        🎵 **Voice-over available in all supported languages!**
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*🎭 Multimodal Cross-Language Poetry AI with Voice-Over - Built with ❤️ using Streamlit*")

if __name__ == "__main__":
    main()