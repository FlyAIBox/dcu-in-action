"""
Test cases for DCU Manager
"""

import pytest
import torch
from common.dcu import DCUManager, DCUInfo


class TestDCUManager:
    """Test DCU Manager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.dcu_manager = DCUManager()
    
    def test_initialization(self):
        """Test DCU manager initialization"""
        assert self.dcu_manager is not None
        assert hasattr(self.dcu_manager, '_is_available')
    
    def test_availability_check(self):
        """Test DCU availability check"""
        is_available = self.dcu_manager.is_available()
        assert isinstance(is_available, bool)
    
    def test_device_count(self):
        """Test getting device count"""
        count = self.dcu_manager.get_device_count()
        assert isinstance(count, int)
        assert count >= 0
    
    def test_memory_info(self):
        """Test getting memory information"""
        memory_info = self.dcu_manager.get_memory_info()
        assert isinstance(memory_info, dict)
        assert 'total' in memory_info
        assert 'used' in memory_info
        assert 'free' in memory_info
        
        # Check if values are non-negative integers
        for key in ['total', 'used', 'free']:
            assert isinstance(memory_info[key], int)
            assert memory_info[key] >= 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="DCU not available")
    def test_device_info_with_dcu(self):
        """Test getting device info when DCU is available"""
        device_info = self.dcu_manager.get_device_info(0)
        
        if device_info is not None:
            assert isinstance(device_info, DCUInfo)
            assert device_info.device_id == 0
            assert isinstance(device_info.name, str)
            assert isinstance(device_info.memory_total, int)
            assert isinstance(device_info.memory_used, int)
            assert isinstance(device_info.memory_free, int)
            assert isinstance(device_info.compute_capability, str)
    
    def test_device_info_without_dcu(self):
        """Test getting device info when DCU is not available"""
        if not self.dcu_manager.is_available():
            device_info = self.dcu_manager.get_device_info(0)
            assert device_info is None
    
    def test_all_devices_info(self):
        """Test getting all devices information"""
        devices = self.dcu_manager.get_all_devices_info()
        assert isinstance(devices, list)
        
        # If DCU is available, check that we get valid device info
        if self.dcu_manager.is_available() and self.dcu_manager.get_device_count() > 0:
            assert len(devices) > 0
            for device in devices:
                assert isinstance(device, DCUInfo)
    
    def test_set_device(self):
        """Test setting DCU device"""
        if self.dcu_manager.is_available() and self.dcu_manager.get_device_count() > 0:
            result = self.dcu_manager.set_device(0)
            assert isinstance(result, bool)
        else:
            # Should return False when no devices available
            result = self.dcu_manager.set_device(0)
            assert result is False
    
    def test_clear_cache(self):
        """Test clearing DCU cache"""
        result = self.dcu_manager.clear_cache()
        assert isinstance(result, bool)
    
    def test_invalid_device_id(self):
        """Test with invalid device ID"""
        device_count = self.dcu_manager.get_device_count()
        invalid_id = device_count + 1
        
        device_info = self.dcu_manager.get_device_info(invalid_id)
        assert device_info is None
        
        result = self.dcu_manager.set_device(invalid_id)
        assert result is False 