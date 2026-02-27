"""
AATE Configuration Management
Centralized settings with environment variable support and validation.
Why: Centralized config prevents hardcoded values, enables environment-specific settings,
and provides runtime validation using Pydantic.
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, validator, Field
from dataclasses import dataclass
from enum import Enum


class TradingMode(str, Enum):
    """Trading operation modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class DataSource(str, Enum):
    """Supported market data sources"""
    CCXT = "ccxt"
    ALPACA = "alpaca"
    COINGECKO = "coingecko"


@dataclass
class ExchangeConfig:
    """Exchange-specific configuration"""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rate_limit: int = 10  # requests per second
    sandbox: bool = True


class Settings(BaseSettings):
    """Main application settings with validation"""
    
    # Application
    APP_NAME: str = "Autonomous Adaptive Trading Engine"
    VERSION: str = "1.0.0"
    TRADING_MODE: TradingMode = TradingMode.PAPER
    LOG_LEVEL: str = "INFO"
    
    # Firebase
    FIREBASE_PROJECT_ID: str = Field(..., env="FIREBASE_PROJECT_ID")
    FIREBASE_CREDENTIALS_PATH: str = Field(..., env="FIREBASE_CREDENTIALS_PATH")
    
    # Data
    DATA_SOURCES: List[DataSource] = [DataSource.CCXT]
    DEFAULT_SYMBOLS: List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    DATA_UPDATE_INTERVAL: int = 60  # seconds
    
    # Trading
    INITIAL_CAPITAL: float = 10000.0
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    STOP_LOSS_PCT: float = 0.02  # 2%
    TAKE_PROFIT_PCT: float = 0.05  # 5%
    
    # RL Configuration
    RL_MODEL_PATH: str = "models/rl_model.zip"
    RL_LEARNING_RATE: float = 0.0003
    RL_GAMMA: float = 0.99
    RL_BUFFER_SIZE: int = 100000
    RL_BATCH_SIZE: int = 64
    
    # Genetic Algorithm
    GA_POPULATION_SIZE: int = 50
    GA_GENERATIONS: int = 100
    GA_MUTATION_RATE: float = 0.1
    GA_CROSSOVER_RATE: float = 0.7
    
    # Risk Management
    MAX_DRAWDOWN_PCT: float = 0.15
    MAX_DAILY_LOSS_PCT: float = 0.05
    CORRELATION_THRESHOLD: float = 0.7
    
    @validator("TRADING_MODE", pre=True)
    def validate_trading_mode(cls, v):
        """Ensure trading mode is valid"""
        if isinstance(v, str):
            v = v.lower()
            if v not in ["backtest", "paper", "live"]:
                raise ValueError(f"Invalid trading mode: {v}")
        return v
    
    @validator("MAX_POSITION_SIZE")
    def validate_position_size(cls, v):
        """Ensure position size is between 0 and 1"""
        if not 0 < v <= 1:
            raise ValueError("MAX_POSITION_SIZE must be between 0 and 1")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Exchange configurations
EXCHANGE_CONFIGS: Dict[str, ExchangeConfig] = {
    "binance": ExchangeConfig(
        name="binance",
        sandbox=True,
        rate_limit=10
    ),
    "coinbase": ExchangeConfig(
        name="coinbase",
        sandbox=True,
        rate_limit=5
    )
}