"""
Audio capture module for system audio recording
"""

import soundcard as sc
import numpy as np
import threading
import queue
import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """Represents a chunk of audio data with metadata"""
    data: np.ndarray
    timestamp: float
    duration: float
    is_speech: bool = False

class SystemAudioCapture:
    """Captures system audio from the default output device (loopback)"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 callback: Optional[Callable] = None):
        """
        Initialize the audio capture system
        
        Args:
            sample_rate: Sample rate in Hz
            chunk_size: Number of samples per chunk
            channels: Number of audio channels
            callback: Callback function for audio chunks
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.callback = callback
        
        self.is_recording = False
        self.audio_thread = None
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Get the default loopback device for system audio
        self.loopback = None
        self.recorder = None
        
        # Audio buffer for continuous recording
        self.audio_buffer = deque(maxlen=int(sample_rate * 60))  # 60 seconds max buffer
        
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize the audio loopback device"""
        try:
            # Get default speakers (loopback)
            default_speaker = sc.default_speaker()
            if default_speaker is None:
                raise RuntimeError("No default speaker found")
            
            self.loopback = sc.get_microphone(id=str(default_speaker.name), 
                                             include_loopback=True)
            
            if self.loopback is None:
                # Fallback to default loopback
                loopbacks = sc.all_microphones(include_loopback=True)
                if loopbacks:
                    self.loopback = loopbacks[0]
                else:
                    raise RuntimeError("No loopback device found")
            
            logger.info(f"Using audio device: {self.loopback.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio device: {e}")
            raise
    
    def start_recording(self):
        """Start recording system audio"""
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.audio_thread.start()
        logger.info("Started audio recording")
    
    def stop_recording(self):
        """Stop recording system audio"""
        if not self.is_recording:
            logger.warning("Not currently recording")
            return
        
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2)
        
        if self.recorder:
            self.recorder.close()
            self.recorder = None
        
        logger.info("Stopped audio recording")
    
    def _record_audio(self):
        """Main recording loop (runs in separate thread)"""
        try:
            with self.loopback.recorder(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size
            ) as recorder:
                self.recorder = recorder
                
                while self.is_recording:
                    try:
                        # Record audio chunk
                        audio_data = recorder.record(numframes=self.chunk_size)
                        
                        # Convert to mono if needed
                        if audio_data.shape[1] > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            audio_data = audio_data.flatten()
                        
                        # Normalize audio
                        audio_data = self._normalize_audio(audio_data)
                        
                        # Create audio chunk
                        chunk = AudioChunk(
                            data=audio_data,
                            timestamp=time.time(),
                            duration=len(audio_data) / self.sample_rate
                        )
                        
                        # Add to queue for processing
                        if not self.audio_queue.full():
                            self.audio_queue.put(chunk)
                        
                        # Callback if provided
                        if self.callback:
                            self.callback(chunk)
                        
                        # Add to buffer
                        self.audio_buffer.extend(audio_data)
                        
                    except Exception as e:
                        if self.is_recording:
                            logger.error(f"Error in recording loop: {e}")
                        break
                
        except Exception as e:
            logger.error(f"Fatal error in audio recording: {e}")
            self.is_recording = False
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio data to [-1, 1] range
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Normalized audio data
        """
        # Avoid division by zero
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """
        Get an audio chunk from the queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            AudioChunk or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_buffer_data(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Get audio data from the buffer
        
        Args:
            duration: Duration in seconds (None for all)
            
        Returns:
            Audio data as numpy array
        """
        if duration:
            num_samples = int(duration * self.sample_rate)
            return np.array(list(self.audio_buffer)[-num_samples:])
        return np.array(list(self.audio_buffer))
    
    def clear_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer.clear()
        
    def get_recording_status(self) -> dict:
        """Get current recording status"""
        return {
            "is_recording": self.is_recording,
            "queue_size": self.audio_queue.qsize(),
            "buffer_size": len(self.audio_buffer),
            "device": self.loopback.name if self.loopback else None
        }

class MicrophoneCapture(SystemAudioCapture):
    """Alternative capture from microphone instead of system audio"""
    
    def _initialize_device(self):
        """Initialize the microphone device"""
        try:
            # Get default microphone
            self.loopback = sc.default_microphone()
            
            if self.loopback is None:
                raise RuntimeError("No default microphone found")
            
            logger.info(f"Using microphone: {self.loopback.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")
            raise
