# -*- coding: utf-8 -*-
"""
Configuration Manager for A.R.A.K System
Handles loading and saving of user and system configurations.
"""

import yaml
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages configuration loading and saving for A.R.A.K system."""
    
    def __init__(self, config_path: str = "src/logic/config.yaml"):
        """Initialize configuration manager with path to config file."""
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            return self._config
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            self._config = self._get_default_config()
            return self._config
        except yaml.YAMLError as e:
            print(f"Error loading config: {e}")
            self._config = self._get_default_config()
            return self._config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to YAML file."""
        if config is not None:
            self._config = config
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get_user_settings(self) -> Dict[str, Any]:
        """Get only user-configurable settings."""
        return self._config.get('exam_policy', {
            'allow_book': False,
            'allow_notebook': False,
            'allow_calculator': False,
            'allow_earphones': False
        })
    
    def update_user_settings(self, settings: Dict[str, bool]) -> bool:
        """Update user settings and save to file."""
        if 'exam_policy' not in self._config:
            self._config['exam_policy'] = {}
        
        # Update only valid user settings
        valid_settings = ['allow_book', 'allow_notebook', 'allow_calculator', 'allow_earphones']
        for key, value in settings.items():
            if key in valid_settings:
                self._config['exam_policy'][key] = bool(value)
        
        return self.save_config()
    
    def get_technical_settings(self) -> Dict[str, Any]:
        """Get all technical settings for the system."""
        config = self._config.copy()
        # Remove user settings to get only technical parameters
        config.pop('exam_policy', None)
        return config
    
    def is_item_allowed(self, item: str) -> bool:
        """Check if a specific item is allowed during the exam."""
        exam_policy = self.get_user_settings()
        return exam_policy.get(f'allow_{item}', False)
    
    def get_detection_weights(self) -> Dict[str, float]:
        """Get optimized detection weights for scoring."""
        weights = self._config.get('weights', {})
        
        # Apply exam policy to weights
        exam_policy = self.get_user_settings()
        
        # If items are allowed, set their weights to 0
        if exam_policy.get('allow_book', False):
            weights['book'] = 0
        if exam_policy.get('allow_calculator', False):
            weights['calculator'] = 0
        if exam_policy.get('allow_notebook', False):
            weights['notebook'] = 0
        if exam_policy.get('allow_earphones', False):
            weights['earphone'] = 0
        
        return weights
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is missing."""
        return {
            'exam_policy': {
                'allow_book': False,
                'allow_notebook': False,
                'allow_calculator': False,
                'allow_earphones': False
            },
            'alert_threshold': 4,
            'phone_conf': 0.50,
            'classes': ['person', 'phone', 'book', 'earphone', 'calculator'],
            'weights': {
                'phone': 8,
                'earphone': 6,
                'person': 7,
                'smartwatch': 6,
                'book': 4,
                'calculator': 4,
                'notebook': 5,
                'gaze_off_per_sec': 1,
                'repetitive_head': 3
            },
            'gaze_duration_threshold': 2.5,
            'repeat_dir_threshold': 4,
            'repeat_window_sec': 12.0,
            'detector_primary': 'yolo11m.pt',
            'detector_secondary': 'models/model_bestV3.pt',
            'detector_conf': 0.40,
            'detector_merge_nms': True,
            'detector_nms_iou': 0.45,
            'detector_merge_mode': 'wbf',
            'class_conf': {
                'phone': 0.50,
                'earphone': 0.45,
                'smartwatch': 0.50,
                'person': 0.55,
                'book': 0.40,
                'calculator': 0.45
            }
        }

# Global configuration manager instance
config_manager = ConfigManager()

def get_config() -> Dict[str, Any]:
    """Get current configuration."""
    return config_manager._config

def get_user_settings() -> Dict[str, Any]:
    """Get user-configurable settings only."""
    return config_manager.get_user_settings()

def update_user_settings(settings: Dict[str, bool]) -> bool:
    """Update user settings."""
    return config_manager.update_user_settings(settings)

def is_item_allowed(item: str) -> bool:
    """Check if item is allowed."""
    return config_manager.is_item_allowed(item)

def get_detection_weights() -> Dict[str, float]:
    """Get detection weights with exam policy applied."""
    return config_manager.get_detection_weights()