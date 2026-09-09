# tests/test_hardware_checker.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from unittest.mock import patch, MagicMock
from hardware_checker import HardwareChecker
import local_models_catalog as catalog


def test_get_system_memory_info():
    info = HardwareChecker.get_system_memory_info()
    assert "total_gb" in info
    assert "available_gb" in info
    assert info["total_gb"] > 0
    assert info["available_gb"] > 0


def test_get_cpu_info():
    info = HardwareChecker.get_cpu_info()
    assert "logical_cores" in info
    assert "physical_cores" in info
    assert "optimal_threads" in info
    assert info["optimal_threads"] >= 1


def test_check_model_compatibility_ok():
    model = catalog.get_model_by_id("qwen-3.5-4b")
    # Mock system with 32 GB RAM, 28 GB available
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 32.0, "available_gb": 28.0}):
        is_safe, msg, status = HardwareChecker.check_model_compatibility(model)
        assert is_safe is True
        assert status == "OK"


def test_check_model_compatibility_warning():
    model = catalog.get_model_by_id("qwen-3.5-4b") # min 4 GB, rec 8 GB
    # Mock system with 16 GB total, 5 GB available
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 16.0, "available_gb": 5.0}):
        is_safe, msg, status = HardwareChecker.check_model_compatibility(model)
        assert is_safe is True
        assert status == "WARNING"


def test_check_model_compatibility_low_free_memory_allows_with_warning():
    model = catalog.get_model_by_id("qwen-3.5-4b") # min 4 GB
    # System has 16 GB total, but only 2 GB free currently -> Should NOT block, should warn
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 16.0, "available_gb": 2.0}):
        is_safe, msg, status = HardwareChecker.check_model_compatibility(model)
        assert is_safe is True
        assert status == "WARNING"
        assert "Düşük Boş Bellek Uyarısı" in msg


def test_check_model_compatibility_blocked_total_ram():
    model = catalog.get_model_by_id("qwen-3.6-35b-moe") # min 24 GB
    # Mock system with only 8 GB total RAM
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 8.0, "available_gb": 6.0}):
        is_safe, msg, status = HardwareChecker.check_model_compatibility(model)
        assert is_safe is False
        assert status == "ERROR"
        assert "engellendi" in msg.lower() or "yetersiz" in msg.lower()


def test_check_file_compatibility_nonexistent():
    is_safe, msg, status = HardwareChecker.check_file_compatibility("/non/existent/model.gguf")
    assert is_safe is False
    assert status == "ERROR"


def test_check_disk_space_success(tmp_path):
    mock_usage = MagicMock()
    mock_usage.free = 50 * (1024 ** 3)
    with patch("psutil.disk_usage", return_value=mock_usage):
        has_space, msg, free_gb = HardwareChecker.check_disk_space(str(tmp_path), 3 * (1024 ** 3))
        assert has_space is True
        assert free_gb == 50.0


def test_check_disk_space_insufficient(tmp_path):
    mock_usage = MagicMock()
    mock_usage.free = 1 * (1024 ** 3)
    with patch("psutil.disk_usage", return_value=mock_usage):
        has_space, msg, free_gb = HardwareChecker.check_disk_space(str(tmp_path), 5 * (1024 ** 3))
        assert has_space is False
        assert free_gb == 1.0
        assert "Yetersiz Disk Alanı" in msg


def test_detect_gpu_devices():
    gpus = HardwareChecker.detect_gpu_devices()
    assert isinstance(gpus, list)
    assert len(gpus) >= 1
    assert all(isinstance(g, str) for g in gpus)


def test_get_acceleration_info_cpu():
    mock_lib = MagicMock(spec=[])
    with patch("llama_cpp.llama_cpp._lib", mock_lib):
        info = HardwareChecker.get_acceleration_info(force_refresh=True)
        assert info["gpu_offload_supported"] is False
        assert "CPU" in info["active_backend"]
        assert "vulkan_installed" in info
        assert "cuda_installed" in info
        assert "detected_gpus" in info


def test_get_acceleration_info_vulkan_backend():
    mock_lib = MagicMock(spec=["ggml_backend_vk_init"])
    with patch("llama_cpp.llama_cpp._lib", mock_lib):
        info = HardwareChecker.get_acceleration_info(force_refresh=True)
        assert info["gpu_offload_supported"] is True
        assert "Vulkan" in info["active_backend"]


def test_get_acceleration_info_cuda_backend():
    mock_lib = MagicMock(spec=["ggml_backend_cuda_init"])
    with patch("llama_cpp.llama_cpp._lib", mock_lib):
        info = HardwareChecker.get_acceleration_info(force_refresh=True)
        assert info["gpu_offload_supported"] is True
        assert "CUDA" in info["active_backend"]



