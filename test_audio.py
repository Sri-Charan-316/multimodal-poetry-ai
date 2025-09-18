import streamlit as st
import tempfile
import os

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    st.success("✅ gTTS is available")
except ImportError:
    TTS_AVAILABLE = False
    st.error("❌ gTTS is not available")

st.title("Audio Test")

if TTS_AVAILABLE:
    text = st.text_input("Enter text to convert to speech:", "Hello, this is a test")
    
    if st.button("Generate Audio") and text:
        try:
            tts = gTTS(text=text, lang='en')
            temp_file = tempfile.mktemp(suffix='.mp3')
            tts.save(temp_file)
            
            if os.path.exists(temp_file):
                st.success(f"Audio created! File size: {os.path.getsize(temp_file)} bytes")
                
                # Create audio player
                with open(temp_file, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
            else:
                st.error("Failed to create audio file")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.error("Please install gTTS: pip install gTTS")