"""
Voice Activity Detection (VAD) and pause detection module
"""

import numpy as np
import webrtcvad
import collections
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal

logger = logging.getLogger(__name__)

@dataclass
class SpeechSegment:
    """Represents a segment of speech with timing information"""
    audio: np.ndarray
    start_time: float
    end_time: float
    duration: float
    confidence: float = 0.0

class VoiceActivityDetector:
    """Advanced VAD with pause detection and speech segmentation"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 aggressiveness: int = 2,
                 frame_duration_ms: int = 30,
                 padding_duration_ms: int = 300,
                 min_speech_duration: float = 0.5,
                 pause_threshold: float = 1.5):
        """
        Initialize the VAD system
        
        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            aggressiveness: VAD aggressiveness (0-3)
            frame_duration_ms: Frame duration (10, 20, or 30 ms)
            padding_duration_ms: Padding around speech
            min_speech_duration: Minimum speech duration in seconds
            pause_threshold: Pause duration to trigger segmentation
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.min_speech_duration = min_speech_duration
        self.pause_threshold = pause_threshold
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Calculate frame size
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.padding_frames = int(padding_duration_ms / frame_duration_ms)
        
        # Ring buffer for smoothing VAD decisions
        self.ring_buffer = collections.deque(maxlen=self.padding_frames)
        
        # Speech state tracking
        self.is_speech = False
        self.speech_start = None
        self.last_speech_time = None
        self.current_segment = []
        self.segments = []
        
        # Energy-based detection parameters
        self.energy_threshold = None
        self.energy_history = collections.deque(maxlen=50)
    
    def process_audio(self, audio_data: np.ndarray) -> List[SpeechSegment]:
        """
        Process audio and detect speech segments with pauses
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            List of completed speech segments
        """
        # Convert to int16 for WebRTC VAD
        audio_int16 = self._convert_to_int16(audio_data)
        
        # Process in frames
        frames = self._create_frames(audio_int16)
        completed_segments = []
        
        current_time = time.time()
        
        for frame in frames:
            # Check if frame is speech
            is_speech_frame = self._is_speech_frame(frame, audio_data)
            
            # Update ring buffer
            self.ring_buffer.append(1 if is_speech_frame else 0)
            
            # Smooth VAD decision
            smoothed_speech = self._smooth_vad_decision()
            
            if smoothed_speech and not self.is_speech:
                # Speech started
                self.is_speech = True
                self.speech_start = current_time
                self.current_segment = [frame]
                logger.debug("Speech started")
                
            elif self.is_speech and smoothed_speech:
                # Continue speech
                self.current_segment.append(frame)
                self.last_speech_time = current_time
                
            elif self.is_speech and not smoothed_speech:
                # Potential end of speech
                self.current_segment.append(frame)
                
                # Check for pause
                if self.last_speech_time and \
                   (current_time - self.last_speech_time) > self.pause_threshold:
                    # Pause detected - complete segment
                    segment = self._complete_segment(current_time)
                    if segment:
                        completed_segments.append(segment)
                        logger.info(f"Completed segment: {segment.duration:.2f}s")
            
            current_time += self.frame_duration_ms / 1000
        
        return completed_segments
    
    def _is_speech_frame(self, frame: bytes, original_audio: np.ndarray) -> bool:
        """
        Determine if a frame contains speech using multiple methods
        
        Args:
            frame: Audio frame as bytes
            original_audio: Original audio data
            
        Returns:
            True if frame contains speech
        """
        # WebRTC VAD
        try:
            vad_result = self.vad.is_speech(frame, self.sample_rate)
        except:
            vad_result = False
        
        # Energy-based detection
        energy = np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16) ** 2))
        self.energy_history.append(energy)
        
        if self.energy_threshold is None and len(self.energy_history) > 10:
            # Set initial threshold based on background noise
            self.energy_threshold = np.mean(self.energy_history) * 1.5
        
        energy_result = energy > self.energy_threshold if self.energy_threshold else False
        
        # Combine both methods
        return vad_result or energy_result
    
    def _smooth_vad_decision(self) -> bool:
        """
        Smooth VAD decisions using ring buffer
        
        Returns:
            Smoothed speech decision
        """
        if not self.ring_buffer:
            return False
        
        # If more than 70% of frames are speech, consider it speech
        speech_ratio = sum(self.ring_buffer) / len(self.ring_buffer)
        return speech_ratio > 0.7
    
    def _complete_segment(self, end_time: float) -> Optional[SpeechSegment]:
        """
        Complete and validate a speech segment
        
        Args:
            end_time: End time of segment
            
        Returns:
            SpeechSegment or None if invalid
        """
        if not self.current_segment or not self.speech_start:
            return None
        
        # Combine frames
        segment_audio = np.concatenate([
            np.frombuffer(frame, dtype=np.int16) 
            for frame in self.current_segment
        ])
        
        duration = end_time - self.speech_start
        
        # Validate segment
        if duration < self.min_speech_duration:
            logger.debug(f"Segment too short: {duration:.2f}s")
            return None
        
        # Calculate confidence based on energy
        confidence = self._calculate_confidence(segment_audio)
        
        segment = SpeechSegment(
            audio=segment_audio,
            start_time=self.speech_start,
            end_time=end_time,
            duration=duration,
            confidence=confidence
        )
        
        # Reset state
        self.is_speech = False
        self.speech_start = None
        self.current_segment = []
        self.last_speech_time = None
        
        return segment
    
    def _calculate_confidence(self, audio: np.ndarray) -> float:
        """
        Calculate speech confidence score
        
        Args:
            audio: Audio segment
            
        Returns:
            Confidence score (0-1)
        """
        if len(audio) == 0:
            return 0.0
        
        # Energy-based confidence
        energy = np.sqrt(np.mean(audio ** 2))
        max_energy = 32768  # Max for int16
        energy_conf = min(energy / max_energy * 2, 1.0)
        
        # Zero-crossing rate (lower for speech)
        zcr = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
        zcr_conf = max(0, 1 - zcr * 10)
        
        # Combined confidence
        return (energy_conf + zcr_conf) / 2
    
    def _convert_to_int16(self, audio: np.ndarray) -> bytes:
        """
        Convert float audio to int16 bytes
        
        Args:
            audio: Audio as float array
            
        Returns:
            Audio as int16 bytes
        """
        # Ensure audio is in [-1, 1] range
        audio = np.clip(audio, -1, 1)
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    def _create_frames(self, audio_bytes: bytes) -> List[bytes]:
        """
        Create fixed-size frames from audio bytes
        
        Args:
            audio_bytes: Audio as bytes
            
        Returns:
            List of frames
        """
        frames = []
        frame_bytes = self.frame_size * 2  # 2 bytes per int16 sample
        
        for i in range(0, len(audio_bytes) - frame_bytes, frame_bytes):
            frames.append(audio_bytes[i:i + frame_bytes])
        
        return frames
    
    def force_complete_segment(self) -> Optional[SpeechSegment]:
        """
        Force completion of current segment (e.g., at max duration)
        
        Returns:
            Completed segment or None
        """
        if self.current_segment and self.speech_start:
            return self._complete_segment(time.time())
        return None
    
    def reset(self):
        """Reset VAD state"""
        self.is_speech = False
        self.speech_start = None
        self.last_speech_time = None
        self.current_segment = []
        self.segments = []
        self.ring_buffer.clear()
        self.energy_history.clear()
        self.energy_threshold = None

class EnhancedVAD(VoiceActivityDetector):
    """Enhanced VAD with additional filtering and processing"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noise_gate_threshold = 0.01
        self.use_spectral_subtraction = True
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio before VAD
        
        Args:
            audio: Input audio
            
        Returns:
            Preprocessed audio
        """
        # Apply noise gate
        audio = self._apply_noise_gate(audio)
        
        # Apply high-pass filter to remove low-frequency noise
        audio = self._apply_highpass_filter(audio)
        
        # Spectral subtraction for noise reduction
        if self.use_spectral_subtraction:
            audio = self._spectral_subtraction(audio)
        
        return audio
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to remove very quiet sounds"""
        mask = np.abs(audio) > self.noise_gate_threshold
        return audio * mask
    
    def _apply_highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise"""
        # Design a high-pass filter (cutoff at 80 Hz for speech)
        nyquist = self.sample_rate / 2
        cutoff = 80 / nyquist
        b, a = signal.butter(5, cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Simple spectral subtraction for noise reduction"""
        # This is a simplified version - real implementation would be more complex
        return audio  # Placeholder for now
