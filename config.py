"""
Configuration settings for the audio transcription application
"""

import os
from pathlib import Path

# Webhook Configuration
WEBHOOK_URL = "https://n8n.smartbytesolutions.co.nz/webhook/interview-audio"
WEBHOOK_TIMEOUT = 30  # seconds
WEBHOOK_RETRY_COUNT = 3
WEBHOOK_RETRY_DELAY = 2  # seconds

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000  # 16kHz for Whisper
AUDIO_CHANNELS = 1  # Mono
AUDIO_CHUNK_SIZE = 1024
AUDIO_FORMAT = "float32"
AUDIO_DEVICE = None  # None for default system audio

# Voice Activity Detection (VAD) Settings
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive filtering
VAD_FRAME_DURATION = 30  # milliseconds (10, 20, or 30)
VAD_PADDING_DURATION = 300  # milliseconds of padding around speech
VAD_MIN_SPEECH_DURATION = 0.5  # minimum seconds of speech to process

# Pause Detection Settings
PAUSE_THRESHOLD = 1.5  # seconds of silence to trigger transcription
MIN_AUDIO_LENGTH = 1.0  # minimum seconds of audio to transcribe
MAX_AUDIO_LENGTH = 30.0  # maximum seconds before forced transcription

# Whisper Configuration
WHISPER_MODEL = "base"  # tiny, base, small, medium, large
WHISPER_LANGUAGE = "en"  # Language code or None for auto-detect
WHISPER_DEVICE = "cpu"  # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float32"  # "float16" for GPU, "float32" for CPU

# System Tray Configuration
APP_NAME = "Audio Transcription Service"
APP_ICON_PATH = Path(__file__).parent / "assets" / "icon.png"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = Path.home() / ".audio_transcription" / "app.log"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Create necessary directories
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Performance Settings
MAX_QUEUE_SIZE = 100  # Maximum audio chunks in processing queue
PROCESSING_THREADS = 2  # Number of transcription threads
