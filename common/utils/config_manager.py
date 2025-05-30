"""
Configuration Manager for handling YAML, JSON, and TOML configs
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path


class ConfigManager:
    """Configuration manager with support for multiple formats"""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize config manager
        
        Args:
            config_file: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = Path(config_file) if config_file else None
        self._config_data = {}
        
        if self.config_file and self.config_file.exists():
            self.load_config()
    
    def load_config(self, config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration from file"""
        if config_file:
            self.config_file = Path(config_file)
        
        if not self.config_file or not self.config_file.exists():
            self.logger.error(f"Config file not found: {self.config_file}")
            return {}
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.yaml' or self.config_file.suffix.lower() == '.yml':
                    self._config_data = yaml.safe_load(f)
                elif self.config_file.suffix.lower() == '.json':
                    self._config_data = json.load(f)
                else:
                    self.logger.error(f"Unsupported config format: {self.config_file.suffix}")
                    return {}
            
            self.logger.info(f"Config loaded from {self.config_file}")
            return self._config_data
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def save_config(self, config_file: Optional[Union[str, Path]] = None) -> bool:
        """Save configuration to file"""
        if config_file:
            self.config_file = Path(config_file)
        
        if not self.config_file:
            self.logger.error("No config file specified")
            return False
        
        try:
            # Create directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self._config_data, f, default_flow_style=False, allow_unicode=True)
                elif self.config_file.suffix.lower() == '.json':
                    json.dump(self._config_data, f, ensure_ascii=False, indent=2)
                else:
                    self.logger.error(f"Unsupported config format: {self.config_file.suffix}")
                    return False
            
            self.logger.info(f"Config saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self._config_data
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary"""
        self._config_data.update(config_dict)
    
    def merge(self, other_config: Union[Dict[str, Any], 'ConfigManager']) -> None:
        """Merge with another configuration"""
        if isinstance(other_config, ConfigManager):
            other_data = other_config._config_data
        else:
            other_data = other_config
        
        self._deep_merge(self._config_data, other_data)
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge two dictionaries"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self._config_data.copy()
    
    def clear(self) -> None:
        """Clear all configuration data"""
        self._config_data.clear()
    
    def has_key(self, key: str) -> bool:
        """Check if key exists (supports dot notation)"""
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return True
        except (KeyError, TypeError):
            return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.get(section, {})
    
    def expand_env_vars(self) -> None:
        """Expand environment variables in config values"""
        self._config_data = self._expand_env_recursive(self._config_data)
    
    def _expand_env_recursive(self, obj: Any) -> Any:
        """Recursively expand environment variables"""
        if isinstance(obj, dict):
            return {k: self._expand_env_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        else:
            return obj 