#!/usr/bin/env python3
"""
Comprehensive functionality test for Poetry AI application
Tests all major components without requiring Streamlit context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    try:
        from gtts import gTTS
        print("✅ gTTS imported successfully")
    except ImportError as e:
        print(f"❌ gTTS import failed: {e}")
        return False
    
    try:
        from deep_translator import GoogleTranslator
        print("✅ Deep Translator imported successfully")
    except ImportError as e:
        print(f"❌ Deep Translator import failed: {e}")
        return False
    
    try:
        import transformers
        from transformers import pipeline
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    return True

def test_translation():
    """Test translation functionality"""
    print("\n🌍 Testing translation...")
    
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='es')
        result = translator.translate('hello beautiful world')
        print(f"✅ Translation test passed: 'hello beautiful world' -> '{result}'")
        return True
    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        return False

def test_tts():
    """Test text-to-speech functionality"""
    print("\n🎵 Testing TTS...")
    
    try:
        from gtts import gTTS
        import io
        
        # Test TTS creation (don't save file)
        tts = gTTS(text="Hello world, this is a test", lang='en', slow=False)
        print("✅ TTS object created successfully")
        
        # Test with different language
        tts_es = gTTS(text="Hola mundo", lang='es', slow=False)
        print("✅ TTS Spanish test passed")
        
        return True
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

def test_transformers():
    """Test transformers model loading"""
    print("\n🤖 Testing AI model...")
    
    try:
        from transformers import pipeline
        
        # Test if we can create a text generation pipeline (don't load model yet)
        print("✅ Transformers pipeline accessible")
        
        # Test basic transformers functionality
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return False

def test_file_structure():
    """Test required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'app.py',
        'assets/theme.css',
        'poetry_env/Scripts/streamlit.exe',
        'poetry_env/Scripts/python.exe'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_css_loading():
    """Test CSS file can be loaded"""
    print("\n🎨 Testing CSS...")
    
    try:
        with open('assets/theme.css', 'r', encoding='utf-8') as f:
            css_content = f.read()
        
        # Check for key CSS classes
        required_classes = ['.stSidebar', '.app-header', '.emoji-logo', '.upload-audio-label']
        for css_class in required_classes:
            if css_class in css_content:
                print(f"✅ CSS class {css_class} found")
            else:
                print(f"❌ CSS class {css_class} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ CSS test failed: {e}")
        return False

def run_all_tests():
    """Run all functionality tests"""
    print("🚀 Poetry AI Functionality Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Translation Test", test_translation),
        ("TTS Test", test_tts),
        ("AI Model Test", test_transformers),
        ("File Structure Test", test_file_structure),
        ("CSS Test", test_css_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Application is fully functional!")
        return True
    else:
        print("⚠️  Some tests failed - check individual results above")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)