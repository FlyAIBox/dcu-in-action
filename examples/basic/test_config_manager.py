#!/usr/bin/env python3
"""
Basic Configuration Manager Test Example

This example demonstrates how to use the Configuration Manager
to handle YAML and JSON configurations.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.utils import ConfigManager


def main():
    """Main function to test Configuration Manager"""
    print("=" * 50)
    print("Configuration Manager Test")
    print("=" * 50)
    
    # Initialize Configuration Manager
    config = ConfigManager()
    
    # Set some configuration values
    config.set('dcu.device_id', 0)
    config.set('dcu.memory_fraction', 0.8)
    config.set('dcu.enable_monitoring', True)
    
    config.set('training.batch_size', 32)
    config.set('training.learning_rate', 0.001)
    config.set('training.epochs', 100)
    config.set('training.model_type', 'llama')
    
    config.set('inference.max_length', 512)
    config.set('inference.temperature', 0.7)
    config.set('inference.top_p', 0.9)
    
    # Test getting values
    print("Configuration Values:")
    print(f"  DCU Device ID: {config.get('dcu.device_id')}")
    print(f"  Memory Fraction: {config.get('dcu.memory_fraction')}")
    print(f"  Enable Monitoring: {config.get('dcu.enable_monitoring')}")
    print(f"  Batch Size: {config.get('training.batch_size')}")
    print(f"  Learning Rate: {config.get('training.learning_rate')}")
    print(f"  Model Type: {config.get('training.model_type')}")
    print(f"  Max Length: {config.get('inference.max_length')}")
    
    # Test getting sections
    print(f"\nDCU Section: {config.get_section('dcu')}")
    print(f"Training Section: {config.get_section('training')}")
    print(f"Inference Section: {config.get_section('inference')}")
    
    # Test checking if key exists
    print(f"\nKey 'dcu.device_id' exists: {config.has_key('dcu.device_id')}")
    print(f"Key 'nonexistent.key' exists: {config.has_key('nonexistent.key')}")
    
    # Test getting with default value
    print(f"Unknown key with default: {config.get('unknown.key', 'default_value')}")
    
    # Save configuration to YAML
    print("\nSaving configuration to YAML...")
    success = config.save_config('configs/test_config.yaml')
    if success:
        print("✓ Configuration saved to configs/test_config.yaml")
    else:
        print("✗ Failed to save configuration")
    
    # Save configuration to JSON
    print("Saving configuration to JSON...")
    success = config.save_config('configs/test_config.json')
    if success:
        print("✓ Configuration saved to configs/test_config.json")
    else:
        print("✗ Failed to save configuration")
    
    # Test loading configuration
    print("\nTesting configuration loading...")
    new_config = ConfigManager('configs/test_config.yaml')
    print(f"Loaded DCU device ID: {new_config.get('dcu.device_id')}")
    print(f"Loaded batch size: {new_config.get('training.batch_size')}")
    
    # Test merging configurations
    print("\nTesting configuration merging...")
    merge_config = ConfigManager()
    merge_config.set('dcu.new_setting', 'new_value')
    merge_config.set('training.batch_size', 64)  # Override existing
    merge_config.set('training.new_param', 'test')
    
    print("Before merge:", new_config.get('training.batch_size'))
    new_config.merge(merge_config)
    print("After merge:", new_config.get('training.batch_size'))
    print("New setting:", new_config.get('dcu.new_setting'))
    print("New param:", new_config.get('training.new_param'))
    
    # Test environment variable expansion
    print("\nTesting environment variable expansion...")
    os.environ['TEST_VAR'] = 'test_value'
    config.set('env.test_path', '$TEST_VAR/path')
    config.expand_env_vars()
    print(f"Expanded path: {config.get('env.test_path')}")
    
    print("\n" + "=" * 50)
    print("Configuration Manager test completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main() 