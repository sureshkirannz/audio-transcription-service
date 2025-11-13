"""
Transcription module using OpenAI Whisper
"""

import whisper
import numpy as np
import logging
import time
import queue
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import tempfile
import os
import wave

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Result from transcription"""
    text: str
    language: str
    confidence: float
    duration: float
    timestamp: float
    segments: List[Dict[str, Any]] = None

class WhisperTranscriber:
    """Whisper-based speech-to-text transcription"""
    
    def __init__(self,
                 model_name: str = "base",
                 language: Optional[str] = "en",
                 device: str = "cpu",
                 compute_type: str = "float32"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            language: Language code or None for auto-detect
            device: Device to use (cpu or cuda)
            compute_type: Compute type (float16 or float32)
        """
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        
        # Load model
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name, device=device)
        logger.info("Whisper model loaded successfully")
        
        # Transcription queue
        self.transcription_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Worker thread
        self.worker_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_transcriptions": 0,
            "successful_transcriptions": 0,
            "failed_transcriptions": 0,
            "empty_transcriptions": 0,
            "total_duration": 0,
            "total_processing_time": 0
        }
    
    def start(self):
        """Start the transcription worker thread"""
        if self.is_running:
            logger.warning("Transcriber already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Transcription worker started")
    
    def stop(self):
        """Stop the transcription worker thread"""
        if not self.is_running:
            logger.warning("Transcriber not running")
            return
        
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Transcription worker stopped")
    
    def transcribe_audio(self, audio: np.ndarray, 
                        sample_rate: int = 16000,
                        async_mode: bool = True) -> Optional[TranscriptionResult]:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio
            async_mode: If True, queue for async processing
            
        Returns:
            TranscriptionResult or None if async
        """
        if async_mode:
            # Queue for async processing
            self.transcription_queue.put({
                "audio": audio,
                "sample_rate": sample_rate,
                "timestamp": time.time()
            })
            return None
        else:
            # Process synchronously
            return self._transcribe(audio, sample_rate)
    
    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> Optional[TranscriptionResult]:
        """
        Internal transcription method
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            TranscriptionResult or None if failed
        """
        start_time = time.time()
        
        try:
            # Validate audio
            if not self._validate_audio(audio):
                self.stats["empty_transcriptions"] += 1
                return None
            
            # Prepare audio for Whisper
            audio_float32 = self._prepare_audio(audio, sample_rate)
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_float32,
                language=self.language,
                task="transcribe",
                fp16=(self.compute_type == "float16"),
                verbose=False
            )
            
            # Process result
            transcription_result = self._process_result(
                result, 
                len(audio) / sample_rate,
                time.time()
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["successful_transcriptions"] += 1
            self.stats["total_duration"] += len(audio) / sample_rate
            self.stats["total_processing_time"] += processing_time
            
            logger.debug(f"Transcription completed in {processing_time:.2f}s")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.stats["failed_transcriptions"] += 1
            return None
        finally:
            self.stats["total_transcriptions"] += 1
    
    def _validate_audio(self, audio: np.ndarray) -> bool:
        """
        Validate audio before transcription
        
        Args:
            audio: Audio data
            
        Returns:
            True if valid
        """
        # Check if audio is empty
        if len(audio) == 0:
            logger.debug("Empty audio")
            return False
        
        # Check if audio is silent (all zeros or very low energy)
        energy = np.sqrt(np.mean(audio ** 2))
        if energy < 0.001:  # Threshold for silence
            logger.debug(f"Audio is silent (energy: {energy:.6f})")
            return False
        
        # Check minimum length (0.5 seconds at 16kHz)
        if len(audio) < 8000:
            logger.debug(f"Audio too short: {len(audio)} samples")
            return False
        
        return True
    
    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Prepare audio for Whisper model
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            Prepared audio
        """
        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure audio is in [-1, 1] range
        audio = np.clip(audio, -1, 1)
        
        # Resample if necessary (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Simple resampling (for better quality, use librosa)
            ratio = 16000 / sample_rate
            new_length = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio
            )
        
        # Pad to minimum length if needed
        min_samples = 3000  # ~0.2 seconds
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        
        return audio
    
    def _process_result(self, result: dict, duration: float, 
                       timestamp: float) -> TranscriptionResult:
        """
        Process Whisper result into TranscriptionResult
        
        Args:
            result: Whisper result dictionary
            duration: Audio duration
            timestamp: Timestamp
            
        Returns:
            TranscriptionResult
        """
        text = result.get("text", "").strip()
        
        # Filter out empty or noise-only transcriptions
        if not text or self._is_noise_transcription(text):
            return None
        
        # Calculate confidence from segments
        segments = result.get("segments", [])
        confidence = self._calculate_confidence(segments)
        
        return TranscriptionResult(
            text=text,
            language=result.get("language", self.language or "unknown"),
            confidence=confidence,
            duration=duration,
            timestamp=timestamp,
            segments=segments
        )
    
    def _is_noise_transcription(self, text: str) -> bool:
        """
        Check if transcription is just noise/artifacts
        
        Args:
            text: Transcribed text
            
        Returns:
            True if noise
        """
        # Common Whisper artifacts for non-speech
        noise_patterns = [
            "thank you", "thanks for watching", "subscribe",
            ".", "..", "...", "♪", "[Music]", "[Applause]",
            "you", ""
        ]
        
        text_lower = text.lower().strip()
        
        # Check for noise patterns
        if text_lower in [p.lower() for p in noise_patterns]:
            return True
        
        # Check if too short (likely noise)
        if len(text_lower) < 3:
            return True
        
        # Check if only punctuation
        if all(c in ".,!?;: " for c in text):
            return True
        
        return False
    
    def _calculate_confidence(self, segments: List[dict]) -> float:
        """
        Calculate average confidence from segments
        
        Args:
            segments: Whisper segments
            
        Returns:
            Average confidence (0-1)
        """
        if not segments:
            return 0.5
        
        confidences = []
        for segment in segments:
            # Whisper provides "no_speech_prob" - invert for confidence
            no_speech_prob = segment.get("no_speech_prob", 0.5)
            confidence = 1 - no_speech_prob
            confidences.append(confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _worker_loop(self):
        """Worker loop for async transcription"""
        logger.info("Transcription worker loop started")
        
        while self.is_running:
            try:
                # Get item from queue with timeout
                item = self.transcription_queue.get(timeout=1)
                
                # Transcribe
                result = self._transcribe(
                    item["audio"],
                    item["sample_rate"]
                )
                
                if result:
                    # Add original timestamp
                    result.timestamp = item["timestamp"]
                    self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def get_result(self, timeout: float = 0.1) -> Optional[TranscriptionResult]:
        """
        Get transcription result from queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            TranscriptionResult or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_statistics(self) -> dict:
        """Get transcription statistics"""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats["successful_transcriptions"] > 0:
            stats["avg_duration"] = stats["total_duration"] / stats["successful_transcriptions"]
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["successful_transcriptions"]
            stats["realtime_factor"] = stats["total_processing_time"] / stats["total_duration"] if stats["total_duration"] > 0 else 0
        else:
            stats["avg_duration"] = 0
            stats["avg_processing_time"] = 0
            stats["realtime_factor"] = 0
        
        return stats

class BatchTranscriber(WhisperTranscriber):
    """Batch processing version of transcriber"""
    
    def __init__(self, batch_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.batch = []
    
    def add_to_batch(self, audio: np.ndarray, sample_rate: int):
        """Add audio to batch for processing"""
        self.batch.append({
            "audio": audio,
            "sample_rate": sample_rate,
            "timestamp": time.time()
        })
        
        if len(self.batch) >= self.batch_size:
            return self.process_batch()
        return []
    
    def process_batch(self) -> List[TranscriptionResult]:
        """Process current batch"""
        results = []
        
        for item in self.batch:
            result = self._transcribe(item["audio"], item["sample_rate"])
            if result:
                result.timestamp = item["timestamp"]
                results.append(result)
        
        self.batch = []
        return results
