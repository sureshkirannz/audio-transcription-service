"""
Webhook client for sending transcriptions
"""

import requests
import json
import time
import logging
import threading
import queue
from typing import Dict, Any, Optional
from dataclasses import asdict
import hashlib

logger = logging.getLogger(__name__)

class WebhookClient:
    """Client for sending transcriptions to webhook endpoint"""
    
    def __init__(self,
                 webhook_url: str,
                 timeout: int = 30,
                 retry_count: int = 3,
                 retry_delay: float = 2.0):
        """
        Initialize webhook client
        
        Args:
            webhook_url: Webhook endpoint URL
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
            retry_delay: Delay between retries
        """
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        # Queue for async sending
        self.send_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        
        # Statistics
        self.stats = {
            "total_sent": 0,
            "successful_sends": 0,
            "failed_sends": 0,
            "duplicate_filtered": 0,
            "last_send_time": None,
            "last_error": None
        }
        
        # Duplicate detection
        self.recent_hashes = queue.Queue(maxsize=100)
        self.hash_set = set()
    
    def start(self):
        """Start the webhook worker thread"""
        if self.is_running:
            logger.warning("Webhook client already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Webhook client started")
    
    def stop(self):
        """Stop the webhook worker thread"""
        if not self.is_running:
            logger.warning("Webhook client not running")
            return
        
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Webhook client stopped")
    
    def send_transcription(self, transcription_result, async_mode: bool = True) -> bool:
        """
        Send transcription to webhook
        
        Args:
            transcription_result: TranscriptionResult object
            async_mode: If True, queue for async sending
            
        Returns:
            Success status (False if async)
        """
        # Convert to dictionary
        if hasattr(transcription_result, '__dict__'):
            data = asdict(transcription_result) if hasattr(transcription_result, '__dataclass_fields__') else transcription_result.__dict__
        else:
            data = transcription_result
        
        # Filter out empty transcriptions
        if not data.get("text") or data.get("text").strip() == "":
            logger.debug("Skipping empty transcription")
            return False
        
        # Check for duplicates
        if self._is_duplicate(data):
            logger.debug("Skipping duplicate transcription")
            self.stats["duplicate_filtered"] += 1
            return False
        
        # Add metadata
        data["sent_at"] = time.time()
        data["client_id"] = "audio_transcription_service"
        
        if async_mode:
            # Queue for async sending
            self.send_queue.put(data)
            return False
        else:
            # Send synchronously
            return self._send(data)
    
    def _is_duplicate(self, data: dict) -> bool:
        """
        Check if transcription is a duplicate
        
        Args:
            data: Transcription data
            
        Returns:
            True if duplicate
        """
        # Create hash of text content
        text = data.get("text", "").strip().lower()
        if not text:
            return False
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check if we've seen this recently
        if text_hash in self.hash_set:
            return True
        
        # Add to recent hashes
        self.hash_set.add(text_hash)
        
        # Maintain size limit
        if self.recent_hashes.full():
            old_hash = self.recent_hashes.get()
            self.hash_set.discard(old_hash)
        
        self.recent_hashes.put(text_hash)
        
        return False
    
    def _send(self, data: dict) -> bool:
        """
        Send data to webhook with retry logic
        
        Args:
            data: Data to send
            
        Returns:
            Success status
        """
        # Validate data
        if not self._validate_data(data):
            logger.warning("Invalid data, skipping send")
            return False
        
        # Prepare payload
        payload = self._prepare_payload(data)
        
        # Send with retries
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=self.timeout,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "AudioTranscriptionService/1.0"
                    }
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully sent transcription: {len(data.get('text', ''))} chars")
                    self.stats["successful_sends"] += 1
                    self.stats["last_send_time"] = time.time()
                    return True
                else:
                    logger.warning(f"Webhook returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Webhook timeout on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                self.stats["last_error"] = str(e)
            
            # Wait before retry
            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        # All attempts failed
        logger.error(f"Failed to send after {self.retry_count} attempts")
        self.stats["failed_sends"] += 1
        return False
    
    def _validate_data(self, data: dict) -> bool:
        """
        Validate data before sending
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid
        """
        # Check required fields
        if "text" not in data:
            return False
        
        # Check text is not empty
        text = data.get("text", "").strip()
        if not text:
            return False
        
        # Check text length (max 10000 chars)
        if len(text) > 10000:
            logger.warning(f"Text too long: {len(text)} chars")
            data["text"] = text[:10000]
        
        return True
    
    def _prepare_payload(self, data: dict) -> dict:
        """
        Prepare webhook payload
        
        Args:
            data: Raw data
            
        Returns:
            Formatted payload
        """
        # Remove None values and segments (can be large)
        cleaned_data = {
            k: v for k, v in data.items() 
            if v is not None and k != "segments"
        }
        
        # Format payload
        payload = {
            "transcription": cleaned_data.get("text"),
            "language": cleaned_data.get("language", "unknown"),
            "confidence": cleaned_data.get("confidence", 0),
            "duration": cleaned_data.get("duration", 0),
            "timestamp": cleaned_data.get("timestamp", time.time()),
            "metadata": {
                "sent_at": cleaned_data.get("sent_at"),
                "client_id": cleaned_data.get("client_id"),
                "session_id": cleaned_data.get("session_id")
            }
        }
        
        return payload
    
    def _worker_loop(self):
        """Worker loop for async webhook sending"""
        logger.info("Webhook worker loop started")
        
        while self.is_running:
            try:
                # Get item from queue with timeout
                data = self.send_queue.get(timeout=1)
                
                # Send to webhook
                self._send(data)
                
                # Update total counter
                self.stats["total_sent"] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def get_statistics(self) -> dict:
        """Get webhook statistics"""
        stats = self.stats.copy()
        
        # Calculate success rate
        if stats["total_sent"] > 0:
            stats["success_rate"] = stats["successful_sends"] / stats["total_sent"]
        else:
            stats["success_rate"] = 0
        
        # Format last send time
        if stats["last_send_time"]:
            stats["time_since_last_send"] = time.time() - stats["last_send_time"]
        
        return stats

class BatchWebhookClient(WebhookClient):
    """Batch sending version of webhook client"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.batch = []
        self.batch_timer = None
    
    def send_transcription(self, transcription_result, async_mode: bool = True) -> bool:
        """Add to batch instead of sending immediately"""
        # Convert and validate
        if hasattr(transcription_result, '__dict__'):
            data = asdict(transcription_result) if hasattr(transcription_result, '__dataclass_fields__') else transcription_result.__dict__
        else:
            data = transcription_result
        
        if not self._validate_data(data) or self._is_duplicate(data):
            return False
        
        # Add to batch
        self.batch.append(data)
        
        # Start timer if not running
        if not self.batch_timer:
            self.batch_timer = threading.Timer(self.batch_timeout, self._send_batch)
            self.batch_timer.start()
        
        # Send if batch is full
        if len(self.batch) >= self.batch_size:
            self._send_batch()
        
        return True
    
    def _send_batch(self):
        """Send current batch"""
        if not self.batch:
            return
        
        # Prepare batch payload
        payload = {
            "batch": True,
            "transcriptions": [self._prepare_payload(d) for d in self.batch],
            "batch_size": len(self.batch),
            "timestamp": time.time()
        }
        
        # Send batch
        success = self._send(payload)
        
        if success:
            logger.info(f"Sent batch of {len(self.batch)} transcriptions")
        
        # Clear batch and timer
        self.batch = []
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
