"""
System tray application for audio transcription service
"""

import pystray
from PIL import Image, ImageDraw
import threading
import logging
import sys
import os
from pathlib import Path
import json
import time

from audio_pipeline import SmartAudioPipeline
import config

logger = logging.getLogger(__name__)

class SystemTrayApp:
    """System tray application for controlling audio transcription"""
    
    def __init__(self):
        """Initialize the system tray application"""
        self.pipeline = None
        self.is_running = False
        self.icon = None
        
        # Settings
        self.use_system_audio = True
        self.settings_file = Path.home() / ".audio_transcription" / "settings.json"
        self.load_settings()
        
        # Create icon
        self.create_icon()
        
        # Statistics
        self.start_time = None
        self.transcription_count = 0
    
    def create_icon(self):
        """Create system tray icon"""
        # Generate icon image
        image = self.generate_icon_image(recording=False)
        
        # Create menu
        menu = pystray.Menu(
            pystray.MenuItem("Start Recording", self.start_recording, default=True),
            pystray.MenuItem("Stop Recording", self.stop_recording),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Audio Source", pystray.Menu(
                pystray.MenuItem(
                    "System Audio",
                    self.set_system_audio,
                    checked=lambda item: self.use_system_audio
                ),
                pystray.MenuItem(
                    "Microphone",
                    self.set_microphone,
                    checked=lambda item: not self.use_system_audio
                )
            )),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Statistics", self.show_statistics),
            pystray.MenuItem("Settings", self.show_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("About", self.show_about),
            pystray.MenuItem("Quit", self.quit_app)
        )
        
        # Create icon
        self.icon = pystray.Icon(
            config.APP_NAME,
            image,
            config.APP_NAME,
            menu
        )
    
    def generate_icon_image(self, recording: bool = False) -> Image:
        """
        Generate icon image
        
        Args:
            recording: Whether currently recording
            
        Returns:
            PIL Image object
        """
        # Create image
        width = 64
        height = 64
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw microphone icon
        mic_color = (255, 0, 0) if recording else (100, 100, 100)
        
        # Microphone body
        draw.ellipse(
            [width//2 - 10, height//4, width//2 + 10, height//2],
            fill=mic_color
        )
        draw.rectangle(
            [width//2 - 10, height//3, width//2 + 10, height//2],
            fill=mic_color
        )
        
        # Microphone stand
        draw.rectangle(
            [width//2 - 2, height//2, width//2 + 2, height//2 + 15],
            fill=mic_color
        )
        draw.rectangle(
            [width//2 - 10, height//2 + 15, width//2 + 10, height//2 + 18],
            fill=mic_color
        )
        
        # Recording indicator
        if recording:
            draw.ellipse(
                [width - 15, 5, width - 5, 15],
                fill=(255, 0, 0)
            )
        
        return image
    
    def start_recording(self, icon, item):
        """Start audio recording"""
        if self.is_running:
            logger.warning("Already recording")
            return
        
        logger.info("Starting recording")
        
        # Initialize pipeline
        self.pipeline = SmartAudioPipeline(
            use_system_audio=self.use_system_audio,
            webhook_url=config.WEBHOOK_URL
        )
        
        # Start pipeline
        try:
            self.pipeline.start()
            self.is_running = True
            self.start_time = time.time()
            
            # Update icon
            self.icon.icon = self.generate_icon_image(recording=True)
            
            # Notify user
            self.icon.notify(
                "Recording Started",
                f"Audio transcription is now active\nSource: {'System Audio' if self.use_system_audio else 'Microphone'}"
            )
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.icon.notify(
                "Error",
                f"Failed to start recording: {str(e)}"
            )
    
    def stop_recording(self, icon, item):
        """Stop audio recording"""
        if not self.is_running:
            logger.warning("Not recording")
            return
        
        logger.info("Stopping recording")
        
        # Stop pipeline
        if self.pipeline:
            stats = self.pipeline.get_statistics()
            self.pipeline.stop()
            self.pipeline = None
        
        self.is_running = False
        
        # Update icon
        self.icon.icon = self.generate_icon_image(recording=False)
        
        # Notify user
        duration = time.time() - self.start_time if self.start_time else 0
        self.icon.notify(
            "Recording Stopped",
            f"Duration: {duration:.1f}s\nTranscriptions: {stats.get('total_transcriptions', 0)}"
        )
    
    def set_system_audio(self, icon, item):
        """Set audio source to system audio"""
        if self.use_system_audio:
            return
        
        self.use_system_audio = True
        self.save_settings()
        
        # Restart if recording
        if self.is_running:
            self.stop_recording(icon, item)
            self.start_recording(icon, item)
    
    def set_microphone(self, icon, item):
        """Set audio source to microphone"""
        if not self.use_system_audio:
            return
        
        self.use_system_audio = False
        self.save_settings()
        
        # Restart if recording
        if self.is_running:
            self.stop_recording(icon, item)
            self.start_recording(icon, item)
    
    def show_statistics(self, icon, item):
        """Show statistics"""
        if not self.pipeline:
            self.icon.notify(
                "Statistics",
                "No active recording session"
            )
            return
        
        stats = self.pipeline.get_statistics()
        
        message = f"""Session: {stats.get('session_id', 'N/A')[:8]}
Runtime: {stats.get('runtime', 0):.1f}s
Audio Processed: {stats.get('total_audio_processed', 0):.1f}s
Transcriptions: {stats.get('total_transcriptions', 0)}
Webhooks Sent: {stats.get('total_webhooks_sent', 0)}
"""
        
        self.icon.notify("Statistics", message)
    
    def show_settings(self, icon, item):
        """Show settings dialog"""
        settings_info = f"""Current Settings:
Webhook: {config.WEBHOOK_URL}
Model: {config.WHISPER_MODEL}
Language: {config.WHISPER_LANGUAGE}
VAD Level: {config.VAD_AGGRESSIVENESS}
Pause Threshold: {config.PAUSE_THRESHOLD}s

Settings file: {self.settings_file}
"""
        
        self.icon.notify("Settings", settings_info)
    
    def show_about(self, icon, item):
        """Show about dialog"""
        about_text = f"""{config.APP_NAME}
Real-time Audio Transcription Service

Captures system audio or microphone input,
transcribes speech to text using Whisper,
and sends to webhook endpoint.

Version: 1.0.0
"""
        
        self.icon.notify("About", about_text)
    
    def quit_app(self, icon, item):
        """Quit the application"""
        logger.info("Quitting application")
        
        # Stop recording if active
        if self.is_running:
            self.stop_recording(icon, item)
        
        # Stop icon
        self.icon.stop()
    
    def load_settings(self):
        """Load settings from file"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.use_system_audio = settings.get('use_system_audio', True)
                    logger.info("Settings loaded")
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
    
    def save_settings(self):
        """Save settings to file"""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump({
                    'use_system_audio': self.use_system_audio
                }, f)
                logger.info("Settings saved")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    def run(self):
        """Run the system tray application"""
        logger.info("Starting system tray application")
        self.icon.run()

def setup_logging():
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create log directory
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=log_format,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    
    # Create and run app
    app = SystemTrayApp()
    
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
