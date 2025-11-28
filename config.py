"""
VTU Final Year Project Configuration
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ProjectConfig:
    """Global project configuration"""
    debug: bool = True
    sample_rate: int = 16000
    max_audio_length: float = 30.0  
    results_dir: str = "results"
    models_dir: str = "models"
    templates_dir: str = "templates"
    static_dir: str = "static"
    
    model_configs: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        self.model_configs = {
            'cnn_lstm': {
                'mfcc_dim': 13,
                'hidden_dim': 128,
                'num_layers': 2,
                'vocab_size': 29,
                'dropout': 0.3
            },
            'wav2vec': {
                'model_name': 'facebook/wav2vec2-base-960h',
                'cache_dir': './models/huggingface_cache'
            }
        }
        self.results_dir = Path(self.results_dir)
        self.models_dir = Path(self.models_dir)
        self.templates_dir = Path(self.templates_dir)
        self.static_dir = Path(self.static_dir)

PROJECT_CONFIG = ProjectConfig()
