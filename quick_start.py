#!/usr/bin/env python3
"""
Quick start script for Audio Transcription Service
Handles installation and initial setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print welcome header"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║        Audio Transcription Service - Quick Start              ║
║           Real-time Speech to Text with Webhooks              ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected.")
        print("   Python 3.8 or higher is required.")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Platform-specific pre-requisites
    system = platform.system()
    
    if system == "Linux":
        print("📦 Installing system packages (may require sudo)...")
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=False)
            subprocess.run(["sudo", "apt-get", "install", "-y", 
                          "portaudio19-dev", "python3-dev"], check=False)
        except:
            print("⚠️  Could not install system packages. Manual installation may be required.")
    
    # Install Python packages
    print("📦 Installing Python packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_whisper_model():
    """Download the default Whisper model"""
    print("\nDownloading Whisper model...")
    print("This may take a few minutes on first run...")
    
    try:
        import whisper
        model = whisper.load_model("base")
        print("✅ Whisper model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download Whisper model: {e}")
        return False

def configure_webhook():
    """Configure webhook URL"""
    print("\nWebhook Configuration")
    print("-" * 40)
    
    default_url = "https://n8n.smartbytesolutions.co.nz/webhook/interview-audio"
    print(f"Default webhook URL: {default_url}")
    
    user_input = input("Enter your webhook URL (press Enter to use default): ").strip()
    
    if user_input:
        # Update config.py with user's webhook
        config_file = Path("config.py")
        if config_file.exists():
            content = config_file.read_text()
            content = content.replace(default_url, user_input)
            config_file.write_text(content)
            print(f"✅ Webhook URL updated to: {user_input}")
        else:
            print("⚠️  Could not update config file")
    else:
        print("✅ Using default webhook URL")
    
    return True

def test_audio_devices():
    """Quick test of audio devices"""
    print("\nTesting audio devices...")
    
    try:
        import soundcard as sc
        
        speakers = sc.all_speakers()
        mics = sc.all_microphones()
        
        print(f"✅ Found {len(speakers)} speaker(s) and {len(mics)} microphone(s)")
        
        if len(speakers) == 0 and len(mics) == 0:
            print("⚠️  No audio devices found. Check your audio drivers.")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️  Could not test audio devices: {e}")
        return True  # Continue anyway

def create_shortcuts():
    """Create desktop/start menu shortcuts"""
    print("\nCreating shortcuts...")
    
    system = platform.system()
    
    if system == "Windows":
        # Create batch file for Windows
        batch_content = f"""@echo off
cd /d "{os.getcwd()}"
python system_tray_app.py
"""
        batch_file = Path("start_transcription.bat")
        batch_file.write_text(batch_content)
        print(f"✅ Created start_transcription.bat")
        
    elif system in ["Linux", "Darwin"]:
        # Create shell script for Unix-like systems
        shell_content = f"""#!/bin/bash
cd "{os.getcwd()}"
python3 system_tray_app.py
"""
        shell_file = Path("start_transcription.sh")
        shell_file.write_text(shell_content)
        shell_file.chmod(0o755)
        print(f"✅ Created start_transcription.sh")
    
    return True

def run_test():
    """Run installation test"""
    print("\nRunning installation test...")
    
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed. Check test_installation.py for details.")
            return False
    except Exception as e:
        print(f"⚠️  Could not run tests: {e}")
        return False

def main():
    """Main quick start process"""
    print_header()
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Download Whisper Model", download_whisper_model),
        ("Configure Webhook", configure_webhook),
        ("Test Audio Devices", test_audio_devices),
        ("Create Shortcuts", create_shortcuts),
        ("Run Tests", run_test),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*60}")
        print(f"Step: {step_name}")
        print(f"{'='*60}")
        
        try:
            if not step_func():
                failed_steps.append(step_name)
                
                # Critical failures
                if step_name in ["Python Version Check", "Install Dependencies"]:
                    print(f"\n❌ Critical failure in: {step_name}")
                    print("Cannot continue with setup.")
                    sys.exit(1)
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            failed_steps.append(step_name)
    
    # Final summary
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    
    if not failed_steps:
        print("""
✅ Installation successful!

To start the application:

1. System Tray Mode (recommended):
   python system_tray_app.py
   
2. Command Line Mode:
   python cli.py --source system

3. Using shortcuts:""")
        
        if platform.system() == "Windows":
            print("   Double-click start_transcription.bat")
        else:
            print("   Run ./start_transcription.sh")
        
        print("""
The app will:
- Capture system audio in the background
- Transcribe speech to text in real-time
- Send transcriptions to your webhook
- Run quietly in the system tray

Right-click the tray icon to control recording.
        """)
    else:
        print(f"\n⚠️  Setup completed with warnings in: {', '.join(failed_steps)}")
        print("\nThe application may still work. Try running:")
        print("  python system_tray_app.py")
        print("\nFor issues, check:")
        print("  - README.md for troubleshooting")
        print("  - test_installation.py for detailed tests")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
