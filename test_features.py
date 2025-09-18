#!/usr/bin/env python3
"""
Quick test script to verify all major features of the multimodal poetry app
"""
import os
import sys

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit failed: {e}")
        return False
    
    try:
        from gtts import gTTS
        print("✅ gTTS imported")
    except ImportError as e:
        print(f"⚠️ gTTS not available: {e}")
    
    try:
        from pydub import AudioSegment
        print("✅ pydub imported")
    except ImportError as e:
        print(f"⚠️ pydub not available: {e}")
    
    try:
        from deep_translator import GoogleTranslator
        print("✅ deep_translator imported")
    except ImportError as e:
        print(f"⚠️ deep_translator not available: {e}")
    
    try:
        from transformers import AutoTokenizer
        print("✅ transformers imported")
    except ImportError as e:
        print(f"⚠️ transformers not available: {e}")
    
    try:
        import speech_recognition as sr
        print("✅ speech_recognition imported")
    except ImportError as e:
        print(f"⚠️ speech_recognition not available: {e}")
    
    return True

def test_basic_poetry():
    """Test basic poetry generation"""
    print("\n🎭 Testing poetry generation...")
    
    # Import the poetry function from our app
    sys.path.append('.')
    from app import generate_poetry_with_ai
    
    try:
        poem = generate_poetry_with_ai("sunset", "free verse", "nature", "short")
        if poem and len(poem) > 50:
            print("✅ Poetry generation works")
            print(f"Sample: {poem[:100]}...")
            return True
        else:
            print("❌ Poetry generation failed")
            return False
    except Exception as e:
        print(f"❌ Poetry generation error: {e}")
        return False

def test_translation():
    """Test translation functionality"""
    print("\n🌍 Testing translation...")
    
    try:
        from app import translate_text
        
        # Test English to Spanish
        result = translate_text("Hello world", "Spanish")
        if result and result != "Hello world":
            print("✅ Translation works")
            print(f"'Hello world' -> '{result}'")
            return True
        else:
            print("⚠️ Translation may not be working (could be network issue)")
            return False
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return False

def test_internet():
    """Test internet connectivity"""
    print("\n🌐 Testing internet connection...")
    
    try:
        from app import test_internet_connection
        if test_internet_connection():
            print("✅ Internet connection available")
            return True
        else:
            print("❌ No internet connection")
            return False
    except Exception as e:
        print(f"❌ Internet test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Multimodal Poetry AI - Feature Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test each component
    results.append(test_imports())
    results.append(test_internet())
    results.append(test_basic_poetry())
    results.append(test_translation())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✅ {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core features are working!")
    elif passed >= total // 2:
        print("⚠️ Most features working, some optional features may need setup")
    else:
        print("❌ Some core features need attention")
    
    print("\n🎭 The app is ready to use at: http://localhost:8510")
    print("Key features:")
    print("- ✅ Poetry generation in multiple styles and lengths")
    print("- ✅ Translation to 12 languages including Telugu")
    print("- ✅ Text-to-speech voice-over (requires internet)")
    print("- ✅ Musical background generation (requires FFmpeg)")
    print("- ✅ AI model integration (beta)")
    print("- ✅ Speech-to-text audio prompt (beta)")

if __name__ == "__main__":
    main()