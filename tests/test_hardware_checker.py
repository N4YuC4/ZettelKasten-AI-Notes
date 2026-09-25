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
    model = catalog.get_model_by_id("gemma-4-e2b")
    # Mock system with 32 GB RAM, 28 GB available
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 32.0, "available_gb": 28.0}):
        is_safe, msg, status = HardwareChecker.check_model_compatibility(model)
        assert is_safe is True
        assert status == "OK"


def test_check_model_compatibility_warning():
    model = catalog.get_model_by_id("gemma-4-e2b") # min 4 GB, rec 8 GB
    # Mock system with 16 GB total, 5 GB available
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 16.0, "available_gb": 5.0}):
        is_safe, msg, status = HardwareChecker.check_model_compatibility(model)
        assert is_safe is True
        assert status == "WARNING"


def test_check_model_compatibility_low_free_memory_allows_with_warning():
    model = catalog.get_model_by_id("gemma-4-e2b") # min 4 GB
    # System has 16 GB total, but only 2 GB free currently -> Should NOT block, should warn
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 16.0, "available_gb": 2.0}):
        is_safe, msg, status = HardwareChecker.check_model_compatibility(model)
        assert is_safe is True
        assert status == "WARNING"
        assert "Low Free Memory Warning" in msg


def test_check_model_compatibility_blocked_total_ram():
    model = catalog.get_model_by_id("gemma-4-26b-moe") # min 16 GB
    # Mock system with only 8 GB total RAM
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 8.0, "available_gb": 6.0}):
        is_safe, msg, status = HardwareChecker.check_model_compatibility(model)
        assert is_safe is False
        assert status == "ERROR"
        assert "blocked" in msg.lower() or "insufficient" in msg.lower()


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
        assert "Insufficient Disk Space" in msg


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


def test_device_performance_profile_detection():
    # Low-end machine (<= 8.5 GB RAM)
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 8.0, "available_gb": 4.0, "used_gb": 4.0, "percent_used": 50.0}):
        with patch.object(HardwareChecker, "get_acceleration_info", return_value={"gpu_offload_supported": False}):
            assert HardwareChecker.get_device_performance_profile() == "LOW_END"

    # High-end machine (>= 24 GB RAM)
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 32.0, "available_gb": 25.0, "used_gb": 7.0, "percent_used": 20.0}):
        with patch.object(HardwareChecker, "get_acceleration_info", return_value={"gpu_offload_supported": True}):
            assert HardwareChecker.get_device_performance_profile() == "HIGH_END"

    # Standard machine (16 GB RAM with GPU)
    with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 16.0, "available_gb": 10.0, "used_gb": 6.0, "percent_used": 40.0}):
        with patch.object(HardwareChecker, "get_acceleration_info", return_value={"gpu_offload_supported": True}):
            assert HardwareChecker.get_device_performance_profile() == "STANDARD"


def test_calculate_adaptive_context_window():
    # 1. Short document: fixed to 16K
    short_text = "Short note text about an idea. " * 50
    ctx_short = HardwareChecker.calculate_adaptive_context_window(short_text)
    assert ctx_short == 16384

    # 2. Medium/Long document on Standard profile: fixed to 16K
    long_text = "Dense academic analysis of neural networks and latent representations. " * 1500
    with patch.object(HardwareChecker, "get_device_performance_profile", return_value="STANDARD"):
        with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 16.0, "available_gb": 10.0}):
            ctx_long = HardwareChecker.calculate_adaptive_context_window(long_text, model_max_ctx=16384)
            assert ctx_long == 16384
            assert ctx_long % 4096 == 0

    # 3. Long document on Low-End profile: fixed to 16K
    with patch.object(HardwareChecker, "get_device_performance_profile", return_value="LOW_END"):
        with patch.object(HardwareChecker, "get_system_memory_info", return_value={"total_gb": 8.0, "available_gb": 3.0}):
            ctx_low_end = HardwareChecker.calculate_adaptive_context_window(long_text, model_max_ctx=16384)
            assert ctx_low_end == 16384




