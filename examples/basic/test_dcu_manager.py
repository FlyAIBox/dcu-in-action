#!/usr/bin/env python3
"""
Basic DCU Manager Test Example

This example demonstrates how to use the DCU Manager to check
device availability and basic information.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.dcu import DCUManager
from common.utils import ConfigManager


def main():
    """Main function to test DCU Manager"""
    print("=" * 50)
    print("DCU Manager Basic Test")
    print("=" * 50)
    
    # Initialize DCU Manager
    dcu = DCUManager()
    
    # Check DCU availability
    print(f"DCU Available: {dcu.is_available()}")
    print(f"Device Count: {dcu.get_device_count()}")
    
    # Get memory information
    memory_info = dcu.get_memory_info()
    print(f"Memory Info: {memory_info}")
    
    # Get all devices information
    devices = dcu.get_all_devices_info()
    print(f"Number of devices found: {len(devices)}")
    
    for i, device in enumerate(devices):
        print(f"\nDevice {i}:")
        print(f"  Name: {device.name}")
        print(f"  Memory Total: {device.memory_total} MB")
        print(f"  Memory Used: {device.memory_used} MB")
        print(f"  Memory Free: {device.memory_free} MB")
        print(f"  Compute Capability: {device.compute_capability}")
    
    # Test configuration manager
    print("\n" + "=" * 50)
    print("Configuration Manager Test")
    print("=" * 50)
    
    config = ConfigManager()
    config.set('dcu.device_id', 0)
    config.set('dcu.memory_fraction', 0.8)
    config.set('training.batch_size', 32)
    
    print(f"DCU Device ID: {config.get('dcu.device_id')}")
    print(f"Memory Fraction: {config.get('dcu.memory_fraction')}")
    print(f"Batch Size: {config.get('training.batch_size')}")
    
    # Save configuration
    config.save_config('test_config.yaml')
    print("Configuration saved to test_config.yaml")
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main() 