"""
Audio processing pipeline combining all components
"""

import numpy as np
import time
import logging
import threading
from typing import Optional, List
from collections import deque
import uuid

from audio_capture import SystemAudioCapture, MicrophoneCapture
from vad_detector import EnhancedVAD, SpeechSegment
from transcription import WhisperTranscriber, TranscriptionResult
from webhook_client import WebhookClient
import config

logger = logging.getLogger(__name__)

class AudioProcessingPipeline:
    """Complete audio processing pipeline from capture to webhook"""
    
    def __init__(self,
                 use_system_audio: bool = True,
                 webhook_url: str = config.WEBHOOK_URL):
        """
        Initialize the audio processing pipeline
        
        Args:
            use_system_audio: If True, capture system audio; else microphone
            webhook_url: Webhook endpoint URL
        """
        self.use_system_audio = use_system_audio
        self.webhook_url = webhook_url
        self.session_id = str(uuid.uuid4())
        
        # Initialize components
        self._initialize_components()
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        
        # Audio buffer for accumulation
        self.audio_buffer = []
        self.buffer_start_time = None
        self.last_speech_time = None
        
        # Statistics
        self.stats = {
            "session_id": self.session_id,
            "start_time": None,
            "total_audio_processed": 0,
            "total_transcriptions": 0,
            "total_webhooks_sent": 0
        }
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        # Audio capture
        if self.use_system_audio:
            self.audio_capture = SystemAudioCapture(
                sample_rate=config.AUDIO_SAMPLE_RATE,
                chunk_size=config.AUDIO_CHUNK_SIZE,
                channels=config.AUDIO_CHANNELS
            )
        else:
            self.audio_capture = MicrophoneCapture(
                sample_rate=config.AUDIO_SAMPLE_RATE,
                chunk_size=config.AUDIO_CHUNK_SIZE,
                channels=config.AUDIO_CHANNELS
            )
        
        # Voice Activity Detection
        self.vad = EnhancedVAD(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            aggressiveness=config.VAD_AGGRESSIVENESS,
            frame_duration_ms=config.VAD_FRAME_DURATION,
            padding_duration_ms=config.VAD_PADDING_DURATION,
            min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
            pause_threshold=config.PAUSE_THRESHOLD
        )
        
        # Transcription
        self.transcriber = WhisperTranscriber(
            model_name=config.WHISPER_MODEL,
            language=config.WHISPER_LANGUAGE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE
        )
        
        # Webhook client
        self.webhook_client = WebhookClient(
            webhook_url=self.webhook_url,
            timeout=config.WEBHOOK_TIMEOUT,
            retry_count=config.WEBHOOK_RETRY_COUNT,
            retry_delay=config.WEBHOOK_RETRY_DELAY
        )
    
    def start(self):
        """Start the audio processing pipeline"""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        logger.info("Starting audio processing pipeline")
        
        # Start components
        self.audio_capture.start_recording()
        self.transcriber.start()
        self.webhook_client.start()
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Update statistics
        self.stats["start_time"] = time.time()
        
        logger.info(f"Pipeline started with session ID: {self.session_id}")
    
    def stop(self):
        """Stop the audio processing pipeline"""
        if not self.is_running:
            logger.warning("Pipeline not running")
            return
        
        logger.info("Stopping audio processing pipeline")
        
        # Stop processing
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Process any remaining audio
        self._process_remaining_audio()
        
        # Stop components
        self.audio_capture.stop_recording()
        self.transcriber.stop()
        self.webhook_client.stop()
        
        logger.info("Pipeline stopped")
    
    def _processing_loop(self):
        """Main processing loop"""
        logger.info("Processing loop started")
        
        while self.is_running:
            try:
                # Get audio chunk from capture
                chunk = self.audio_capture.get_audio_chunk(timeout=0.1)
                
                if chunk:
                    # Process audio chunk
                    self._process_audio_chunk(chunk)
                
                # Check for transcription results
                self._check_transcription_results()
                
                # Check for timeout (force transcription after MAX_AUDIO_LENGTH)
                self._check_buffer_timeout()
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
    
    def _process_audio_chunk(self, chunk):
        """
        Process a single audio chunk
        
        Args:
            chunk: AudioChunk object
        """
        # Preprocess audio
        audio_data = self.vad.preprocess_audio(chunk.data)
        
        # Add to buffer
        if self.buffer_start_time is None:
            self.buffer_start_time = chunk.timestamp
        self.audio_buffer.append(audio_data)
        
        # Update statistics
        self.stats["total_audio_processed"] += chunk.duration
        
        # Process with VAD
        segments = self.vad.process_audio(audio_data)
        
        # Process completed segments
        for segment in segments:
            self._process_speech_segment(segment)
    
    def _process_speech_segment(self, segment: SpeechSegment):
        """
        Process a completed speech segment
        
        Args:
            segment: SpeechSegment object
        """
        logger.info(f"Processing speech segment: {segment.duration:.2f}s, confidence: {segment.confidence:.2f}")
        
        # Filter low-confidence segments
        if segment.confidence < 0.3:
            logger.debug(f"Skipping low-confidence segment: {segment.confidence:.2f}")
            return
        
        # Send to transcriber
        self.transcriber.transcribe_audio(
            segment.audio,
            sample_rate=config.AUDIO_SAMPLE_RATE,
            async_mode=True
        )
        
        # Update last speech time
        self.last_speech_time = segment.end_time
        
        # Clear processed audio from buffer
        self._clear_buffer()
    
    def _check_transcription_results(self):
        """Check for and process transcription results"""
        result = self.transcriber.get_result(timeout=0.01)
        
        if result:
            self._process_transcription_result(result)
    
    def _process_transcription_result(self, result: TranscriptionResult):
        """
        Process a transcription result
        
        Args:
            result: TranscriptionResult object
        """
        logger.info(f"Transcription: '{result.text[:50]}...' (confidence: {result.confidence:.2f})")
        
        # Add session ID
        result_dict = {
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "duration": result.duration,
            "timestamp": result.timestamp,
            "session_id": self.session_id
        }
        
        # Send to webhook
        self.webhook_client.send_transcription(result_dict, async_mode=True)
        
        # Update statistics
        self.stats["total_transcriptions"] += 1
        self.stats["total_webhooks_sent"] += 1
    
    def _check_buffer_timeout(self):
        """Check if buffer has exceeded maximum duration"""
        if not self.audio_buffer or not self.buffer_start_time:
            return
        
        buffer_duration = time.time() - self.buffer_start_time
        
        # Force transcription if buffer is too long
        if buffer_duration > config.MAX_AUDIO_LENGTH:
            logger.info(f"Buffer timeout: forcing transcription after {buffer_duration:.2f}s")
            self._force_transcription()
        
        # Also check for pause timeout
        elif self.last_speech_time:
            pause_duration = time.time() - self.last_speech_time
            if pause_duration > config.PAUSE_THRESHOLD:
                logger.info(f"Pause detected: {pause_duration:.2f}s")
                self._force_transcription()
    
    def _force_transcription(self):
        """Force transcription of current buffer"""
        if not self.audio_buffer:
            return
        
        # Combine buffer audio
        audio_data = np.concatenate(self.audio_buffer)
        
        # Check if it's worth transcribing
        if len(audio_data) / config.AUDIO_SAMPLE_RATE < config.MIN_AUDIO_LENGTH:
            logger.debug("Buffer too short for transcription")
            self._clear_buffer()
            return
        
        # Force complete current VAD segment
        segment = self.vad.force_complete_segment()
        if segment:
            self._process_speech_segment(segment)
        else:
            # Create segment from buffer
            segment = SpeechSegment(
                audio=audio_data,
                start_time=self.buffer_start_time,
                end_time=time.time(),
                duration=len(audio_data) / config.AUDIO_SAMPLE_RATE,
                confidence=0.5
            )
            self._process_speech_segment(segment)
        
        # Clear buffer
        self._clear_buffer()
    
    def _clear_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer = []
        self.buffer_start_time = None
        self.vad.reset()
    
    def _process_remaining_audio(self):
        """Process any remaining audio when stopping"""
        logger.info("Processing remaining audio")
        
        # Force transcription of buffer
        if self.audio_buffer:
            self._force_transcription()
        
        # Wait for pending transcriptions
        time.sleep(2)
        
        # Process any remaining results
        while True:
            result = self.transcriber.get_result(timeout=0.1)
            if not result:
                break
            self._process_transcription_result(result)
    
    def get_statistics(self) -> dict:
        """Get pipeline statistics"""
        stats = self.stats.copy()
        
        # Add component statistics
        stats["audio_capture"] = self.audio_capture.get_recording_status()
        stats["transcriber"] = self.transcriber.get_statistics()
        stats["webhook"] = self.webhook_client.get_statistics()
        
        # Calculate runtime
        if stats["start_time"]:
            stats["runtime"] = time.time() - stats["start_time"]
        
        return stats

class SmartAudioPipeline(AudioProcessingPipeline):
    """Enhanced pipeline with smart filtering and optimization"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Smart filtering thresholds
        self.min_words = 3  # Minimum words for valid transcription
        self.min_confidence = 0.5  # Minimum confidence threshold
        
        # Deduplication
        self.recent_transcriptions = deque(maxlen=10)
    
    def _process_transcription_result(self, result: TranscriptionResult):
        """Enhanced processing with smart filtering"""
        
        # Filter by word count
        word_count = len(result.text.split())
        if word_count < self.min_words:
            logger.debug(f"Filtered: too few words ({word_count})")
            return
        
        # Filter by confidence
        if result.confidence < self.min_confidence:
            logger.debug(f"Filtered: low confidence ({result.confidence:.2f})")
            return
        
        # Check for duplicates
        if self._is_duplicate_transcription(result.text):
            logger.debug("Filtered: duplicate transcription")
            return
        
        # Process normally
        super()._process_transcription_result(result)
        
        # Add to recent transcriptions
        self.recent_transcriptions.append(result.text)
    
    def _is_duplicate_transcription(self, text: str) -> bool:
        """Check if transcription is a duplicate"""
        # Simple similarity check
        for recent in self.recent_transcriptions:
            if self._text_similarity(text, recent) > 0.8:
                return True
        return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
