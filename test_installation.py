#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import time
import numpy as np
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    modules = [
        ("soundcard", "Audio capture library"),
        ("whisper", "OpenAI Whisper"),
        ("webrtcvad", "Voice Activity Detection"),
        ("pystray", "System tray support"),
        ("requests", "HTTP client"),
        ("scipy", "Scientific computing"),
        ("PIL", "Image processing"),
    ]
    
    failed = []
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:15} - {description}")
        except ImportError as e:
            print(f"✗ {module_name:15} - {description} - Error: {e}")
            failed.append(module_name)
    
    return len(failed) == 0

def test_audio_devices():
    """Test audio device detection"""
    print("\nTesting audio devices...")
    
    try:
        import soundcard as sc
        
        # List all speakers
        speakers = sc.all_speakers()
        print(f"Found {len(speakers)} speaker device(s):")
        for speaker in speakers:
            print(f"  - {speaker.name}")
        
        # List all microphones
        mics = sc.all_microphones()
        print(f"\nFound {len(mics)} microphone device(s):")
        for mic in mics:
            print(f"  - {mic.name}")
        
        # Check for loopback devices
        loopbacks = sc.all_microphones(include_loopback=True)
        loopback_count = len(loopbacks) - len(mics)
        print(f"\nFound {loopback_count} loopback device(s)")
        
        return len(speakers) > 0 or len(mics) > 0
        
    except Exception as e:
        print(f"Error testing audio devices: {e}")
        return False

def test_whisper_model():
    """Test Whisper model loading"""
    print("\nTesting Whisper model...")
    
    try:
        import whisper
        
        print("Loading Whisper 'tiny' model (smallest for testing)...")
        model = whisper.load_model("tiny")
        print("✓ Whisper model loaded successfully")
        
        # Test with dummy audio
        print("Testing transcription with dummy audio...")
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = model.transcribe(dummy_audio)
        print("✓ Whisper transcription test completed")
        
        return True
        
    except Exception as e:
        print(f"Error testing Whisper: {e}")
        return False

def test_vad():
    """Test Voice Activity Detection"""
    print("\nTesting Voice Activity Detection...")
    
    try:
        import webrtcvad
        
        vad = webrtcvad.Vad(2)
        
        # Create test frame (30ms of audio at 16kHz)
        frame_size = int(16000 * 0.03)
        silence = np.zeros(frame_size, dtype=np.int16).tobytes()
        
        is_speech = vad.is_speech(silence, 16000)
        print(f"✓ VAD test completed (silence detected as speech: {is_speech})")
        
        return True
        
    except Exception as e:
        print(f"Error testing VAD: {e}")
        return False

def test_webhook():
    """Test webhook connectivity"""
    print("\nTesting webhook connectivity...")
    
    try:
        import requests
        from config import WEBHOOK_URL
        
        print(f"Testing connection to: {WEBHOOK_URL}")
        
        # Try OPTIONS request first (less intrusive)
        try:
            response = requests.options(WEBHOOK_URL, timeout=5)
            print(f"✓ Webhook responded with status: {response.status_code}")
            return True
        except:
            # Try GET as fallback
            response = requests.get(WEBHOOK_URL, timeout=5)
            print(f"✓ Webhook responded with status: {response.status_code}")
            return True
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to webhook (connection refused)")
        return False
    except requests.exceptions.Timeout:
        print("✗ Webhook connection timeout")
        return False
    except Exception as e:
        print(f"✗ Error testing webhook: {e}")
        return False

def test_pipeline():
    """Test the audio pipeline initialization"""
    print("\nTesting audio pipeline...")
    
    try:
        from audio_pipeline import SmartAudioPipeline
        
        print("Initializing pipeline (not starting capture)...")
        pipeline = SmartAudioPipeline(use_system_audio=False)
        print("✓ Pipeline initialized successfully")
        
        # Get initial statistics
        stats = pipeline.get_statistics()
        print(f"✓ Pipeline statistics accessible: Session ID = {stats['session_id'][:8]}...")
        
        return True
        
    except Exception as e:
        print(f"Error testing pipeline: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Audio Transcription Service - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Audio Devices", test_audio_devices),
        ("Whisper Model", test_whisper_model),
        ("Voice Activity Detection", test_vad),
        ("Webhook Connectivity", test_webhook),
        ("Pipeline Initialization", test_pipeline),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\nError running {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("-" * 60)
    
    passed = 0
    failed = 0
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name:30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n✅ All tests passed! The application is ready to use.")
        print("\nTo start the application:")
        print("  GUI Mode:  python system_tray_app.py")
        print("  CLI Mode:  python cli.py --source system")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("You may need to:")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Check audio device permissions")
        print("  - Verify webhook URL is accessible")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
