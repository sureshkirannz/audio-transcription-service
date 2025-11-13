"""
Command-line interface for audio transcription service
"""

import argparse
import signal
import sys
import time
import logging
from pathlib import Path

from audio_pipeline import SmartAudioPipeline
import config

logger = logging.getLogger(__name__)

class CLIApp:
    """Command-line interface for audio transcription"""
    
    def __init__(self, args):
        """
        Initialize CLI app
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.pipeline = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the transcription service"""
        logger.info("Starting audio transcription service")
        
        # Override config with CLI arguments
        if self.args.webhook:
            config.WEBHOOK_URL = self.args.webhook
        if self.args.model:
            config.WHISPER_MODEL = self.args.model
        if self.args.language:
            config.WHISPER_LANGUAGE = self.args.language
        
        # Initialize pipeline
        self.pipeline = SmartAudioPipeline(
            use_system_audio=(self.args.source == "system"),
            webhook_url=config.WEBHOOK_URL
        )
        
        # Start pipeline
        try:
            self.pipeline.start()
            self.running = True
            
            print(f"""
╔══════════════════════════════════════════════════════════╗
║          Audio Transcription Service Started              ║
╠══════════════════════════════════════════════════════════╣
║ Source:   {self.args.source:48} ║
║ Model:    {config.WHISPER_MODEL:48} ║
║ Language: {config.WHISPER_LANGUAGE or 'auto':48} ║
║ Webhook:  {config.WEBHOOK_URL[:48]:48} ║
╠══════════════════════════════════════════════════════════╣
║ Press Ctrl+C to stop                                      ║
╚══════════════════════════════════════════════════════════╝
""")
            
            # Monitor loop
            self.monitor()
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            print(f"Error: {e}")
            sys.exit(1)
    
    def stop(self):
        """Stop the transcription service"""
        if not self.running:
            return
        
        logger.info("Stopping audio transcription service")
        
        if self.pipeline:
            # Get final statistics
            stats = self.pipeline.get_statistics()
            
            # Stop pipeline
            self.pipeline.stop()
            
            # Print summary
            print(f"""
╔══════════════════════════════════════════════════════════╗
║                    Session Summary                        ║
╠══════════════════════════════════════════════════════════╣
║ Runtime:        {stats.get('runtime', 0):>10.1f} seconds                      ║
║ Audio Processed:{stats.get('total_audio_processed', 0):>10.1f} seconds                      ║
║ Transcriptions: {stats.get('total_transcriptions', 0):>10}                               ║
║ Webhooks Sent:  {stats.get('total_webhooks_sent', 0):>10}                               ║
╚══════════════════════════════════════════════════════════╝
""")
        
        self.running = False
    
    def monitor(self):
        """Monitor the service and display statistics"""
        last_stats_time = 0
        
        while self.running:
            try:
                time.sleep(1)
                
                # Display statistics every 30 seconds if verbose
                if self.args.verbose and time.time() - last_stats_time > 30:
                    self.print_statistics()
                    last_stats_time = time.time()
                
            except KeyboardInterrupt:
                break
    
    def print_statistics(self):
        """Print current statistics"""
        if not self.pipeline:
            return
        
        stats = self.pipeline.get_statistics()
        
        print(f"""
-------------------- Statistics --------------------
Runtime:         {stats.get('runtime', 0):.1f}s
Audio Processed: {stats.get('total_audio_processed', 0):.1f}s
Transcriptions:  {stats.get('total_transcriptions', 0)}
Webhooks Sent:   {stats.get('total_webhooks_sent', 0)}
""")
        
        # Transcriber stats
        trans_stats = stats.get('transcriber', {})
        if trans_stats.get('successful_transcriptions', 0) > 0:
            print(f"""Transcriber:
  Success Rate:  {trans_stats.get('successful_transcriptions', 0)}/{trans_stats.get('total_transcriptions', 0)}
  Avg Duration:  {trans_stats.get('avg_duration', 0):.2f}s
  Realtime Factor: {trans_stats.get('realtime_factor', 0):.2f}x
""")
        
        # Webhook stats
        webhook_stats = stats.get('webhook', {})
        if webhook_stats.get('total_sent', 0) > 0:
            print(f"""Webhook:
  Success Rate:  {webhook_stats.get('success_rate', 0):.1%}
  Duplicates Filtered: {webhook_stats.get('duplicate_filtered', 0)}
""")
        
        print("-" * 52)

def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Audio Transcription Service - Real-time speech to text with webhook integration"
    )
    
    # Audio source
    parser.add_argument(
        "--source",
        choices=["system", "microphone"],
        default="system",
        help="Audio source to capture (default: system)"
    )
    
    # Webhook
    parser.add_argument(
        "--webhook",
        type=str,
        help=f"Webhook URL (default: {config.WEBHOOK_URL})"
    )
    
    # Whisper settings
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model to use (default: {config.WHISPER_MODEL})"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        help="Language code (e.g., en, es, fr) or leave empty for auto-detect"
    )
    
    # VAD settings
    parser.add_argument(
        "--vad-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=config.VAD_AGGRESSIVENESS,
        help="VAD aggressiveness level (0-3, default: 2)"
    )
    
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=config.PAUSE_THRESHOLD,
        help="Pause threshold in seconds (default: 1.5)"
    )
    
    # Output settings
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with statistics"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=config.LOG_LEVEL,
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        default=config.LOG_FILE,
        help="Log file path"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create log directory
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout) if args.verbose else logging.NullHandler()
        ]
    )
    
    # Update config with arguments
    if args.vad_level is not None:
        config.VAD_AGGRESSIVENESS = args.vad_level
    if args.pause_threshold is not None:
        config.PAUSE_THRESHOLD = args.pause_threshold
    
    # Create and run app
    app = CLIApp(args)
    
    try:
        app.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        app.stop()

if __name__ == "__main__":
    main()
