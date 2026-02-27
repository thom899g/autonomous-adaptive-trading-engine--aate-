"""
Advanced logging system with structured JSON output and Firebase integration.
Why: Structured logging enables better log analysis and monitoring. Firebase integration
allows real-time log viewing across distributed systems.
"""
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import traceback

from firebase_admin import firestore
from config.settings import settings


class LogLevel(Enum):
    """Standard log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """Advanced logger with structured JSON output and Firebase integration"""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = getattr(logging, level.upper())
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Firebase Firestore client (lazy initialization)
        self._firestore_client = None
        
        # Cache logs to batch write to Firebase
        self._log_cache = []
        self._cache_size = 100
        
    @property
    def firestore_client(self):
        """Lazy initialization of Firestore client"""
        if self._firestore_client