# hardware_checker.py
#
# Hardware resource inspection and safety validation.
# Prevents users from launching models that exceed system RAM/VRAM capacity or disk storage.

import os
from env_config import configure_headless_environment

configure_headless_environment()

import sys
import glob
import shutil
import subprocess
import ctypes.util
import struct
import psutil
from typing import Tuple, Dict, Any, Optional, List
from local_models_catalog import LocalModelInfo, get_model_by_id
from logger import log_debug, log_error


class HardwareChecker:
    """
    Inspects host hardware resources (RAM, CPU cores, Disk space) and calculates
    safe execution boundaries for local GGUF models.
    """

    @staticmethod
    def get_system_memory_info() -> Dict[str, float]:
        """
        Returns system RAM statistics in Gigabytes (GB).
        """
        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "available_gb": round(vm.available / (1024 ** 3), 2),
            "used_gb": round(vm.used / (1024 ** 3), 2),
            "percent_used": vm.percent,
        }

    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """
        Returns CPU core count for optimal threading.
        """
        logical_cores = psutil.cpu_count(logical=True) or 4
        physical_cores = psutil.cpu_count(logical=False) or logical_cores
        # Optimal thread count: usually physical cores minus 1 for UI responsiveness
        optimal_threads = max(1, physical_cores - 1)
        return {
            "logical_cores": logical_cores,
            "physical_cores": physical_cores,
            "optimal_threads": optimal_threads,
        }

    @staticmethod
    def detect_gpu_devices() -> List[str]:
        """
        Detects installed graphics hardware (AMD, Intel, NVIDIA, Apple Silicon).
        Returns a list of human-readable GPU names.
        """
        gpus: List[str] = []
        plat = sys.platform

        if plat == "darwin":
            return ["Apple Silicon / Metal GPU"]

        if plat.startswith("linux"):
            if shutil.which("lspci"):
                try:
                    out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
                    for line in out.splitlines():
                        if any(t in line for t in ["VGA compatible controller:", "3D controller:", "Display controller:"]):
                            part = line.split("controller:", 1)[-1].strip()
                            if " (rev " in part:
                                part = part.split(" (rev ")[0].strip()
                            if part and part not in gpus:
                                gpus.append(part)
                except Exception:
                    pass

            if not gpus:
                vendor_map = {
                    "0x10de": "NVIDIA GPU",
                    "0x1002": "AMD Radeon GPU",
                    "0x8086": "Intel Graphics",
                }
                for f in glob.glob("/sys/class/drm/card*/device/vendor"):
                    try:
                        with open(f, "r") as vf:
                            v = vf.read().strip().lower()
                            name = vendor_map.get(v)
                            if name and name not in gpus:
                                gpus.append(name)
                    except Exception:
                        pass

        elif plat == "win32":
            try:
                out = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                for line in out.splitlines()[1:]:
                    line = line.strip()
                    if line and line not in gpus:
                        gpus.append(line)
            except Exception:
                pass

        return gpus or ["Standart / Tümleşik Grafik"]

    _cached_acceleration_info: Optional[Dict[str, Any]] = None
    _cached_optimal_vulkan_device: Optional[int] = None

    @staticmethod
    def get_optimal_vulkan_device_index() -> Optional[int]:
        """
        Scans available Vulkan physical devices and returns the index of the discrete GPU
        (e.g., dedicated NVIDIA or AMD dGPU) when present.
        On hybrid graphics laptops (Intel iGPU + NVIDIA dGPU), this prevents the headless
        compute backend from accidentally picking the integrated display GPU, which deadlocks
        with the Wayland display compositor.
        """
        if HardwareChecker._cached_optimal_vulkan_device is not None:
            return HardwareChecker._cached_optimal_vulkan_device

        try:
            vulkan_lib = None
            for name in ["libvulkan.so.1", "libvulkan.so", "vulkan-1.dll"]:
                try:
                    vulkan_lib = ctypes.CDLL(name)
                    break
                except Exception:
                    pass
            if not vulkan_lib:
                return None

            class VkApplicationInfo(ctypes.Structure):
                _fields_ = [
                    ("sType", ctypes.c_uint32),
                    ("pNext", ctypes.c_void_p),
                    ("pApplicationName", ctypes.c_char_p),
                    ("applicationVersion", ctypes.c_uint32),
                    ("pEngineName", ctypes.c_char_p),
                    ("engineVersion", ctypes.c_uint32),
                    ("apiVersion", ctypes.c_uint32),
                ]

            class VkInstanceCreateInfo(ctypes.Structure):
                _fields_ = [
                    ("sType", ctypes.c_uint32),
                    ("pNext", ctypes.c_void_p),
                    ("flags", ctypes.c_uint32),
                    ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
                    ("enabledLayerCount", ctypes.c_uint32),
                    ("ppEnabledLayerNames", ctypes.c_void_p),
                    ("enabledExtensionCount", ctypes.c_uint32),
                    ("ppEnabledExtensionNames", ctypes.c_void_p),
                ]

            app_info = VkApplicationInfo(
                sType=0,
                pNext=None,
                pApplicationName=b"probe",
                applicationVersion=1,
                pEngineName=b"probe",
                engineVersion=1,
                apiVersion=(1 << 22) | (2 << 12),
            )

            create_info = VkInstanceCreateInfo(
                sType=1,
                pNext=None,
                flags=0,
                pApplicationInfo=ctypes.pointer(app_info),
                enabledLayerCount=0,
                ppEnabledLayerNames=None,
                enabledExtensionCount=0,
                ppEnabledExtensionNames=None,
            )

            instance = ctypes.c_void_p()
            if vulkan_lib.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance)) != 0:
                return None

            count = ctypes.c_uint32(0)
            vulkan_lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
            if count.value == 0:
                vulkan_lib.vkDestroyInstance(instance, None)
                return None

            devices = (ctypes.c_void_p * count.value)()
            vulkan_lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), devices)

            discrete_idx = None
            integrated_idx = None

            for i in range(count.value):
                # Vulkan specification requires VkPhysicalDeviceProperties buffer (~824+ bytes).
                # Using a 4096-byte buffer avoids heap corruption from struct truncation.
                buf = ctypes.create_string_buffer(4096)
                vulkan_lib.vkGetPhysicalDeviceProperties(devices[i], buf)
                # First 20 bytes: apiVersion (4), driverVersion (4), vendorID (4), deviceID (4), deviceType (4)
                _, _, _, _, device_type = struct.unpack_from("<IIIII", buf.raw, 0)
                # 2 = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 1 = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                if device_type == 2 and discrete_idx is None:
                    discrete_idx = i
                elif device_type == 1 and integrated_idx is None:
                    integrated_idx = i

            vulkan_lib.vkDestroyInstance(instance, None)
            best_idx = discrete_idx if discrete_idx is not None else integrated_idx
            HardwareChecker._cached_optimal_vulkan_device = best_idx
            return best_idx
        except Exception as e:
            log_debug(f"HardwareChecker Vulkan device probe error: {e}")
            return None

    @staticmethod
    def configure_vulkan_environment(enable_gpu: bool = True) -> None:
        """
        Applies process-level environment variables to prevent Vulkan layer injection,
        Optimus presentation deadlocks, and directs llama.cpp to the dedicated GPU.
        """
        os.environ["VK_LOADER_LAYERS_DISABLE"] = "*"
        os.environ["DISABLE_LAYER_NV_OPTIMUS_1"] = "1"
        os.environ["DISABLE_LAYER_NV_PRESENT_1"] = "1"

        if enable_gpu:
            best_dev = HardwareChecker.get_optimal_vulkan_device_index()
            if best_dev is not None:
                os.environ["GGML_VK_VISIBLE_DEVICES"] = str(best_dev)
                log_debug(f"Configured GGML_VK_VISIBLE_DEVICES={best_dev} for optimal discrete GPU compute.")
        else:
            os.environ["GGML_VK_VISIBLE_DEVICES"] = "-1"

    @staticmethod
    def get_acceleration_info(force_refresh: bool = False) -> Dict[str, Any]:
        """
        Detects hardware acceleration capabilities (Vulkan, CUDA, Metal, CPU).
        Safely checks compiled backend symbols on llama-cpp-python without executing
        crash-prone C++ device enumeration, and caches results to prevent UI freezes.
        """
        if HardwareChecker._cached_acceleration_info is not None and not force_refresh:
            return HardwareChecker._cached_acceleration_info

        gpus = HardwareChecker.detect_gpu_devices()
        gpu_str = ", ".join(gpus)

        vulkan_installed = bool(ctypes.util.find_library("vulkan") or ctypes.util.find_library("vulkan-1"))
        cuda_installed = bool(ctypes.util.find_library("cuda") or ctypes.util.find_library("nvcuda"))

        gpu_offload_supported = False
        active_backend = "CPU (Standart)"

        try:
            import llama_cpp
            lib = getattr(getattr(llama_cpp, "llama_cpp", None), "_lib", None)
            if lib:
                has_vk = getattr(lib, "ggml_backend_vk_init", None) is not None or getattr(lib, "ggml_backend_vulkan_init", None) is not None
                has_cuda = getattr(lib, "ggml_backend_cuda_init", None) is not None or getattr(lib, "ggml_cuda_init", None) is not None
                has_metal = getattr(lib, "ggml_backend_metal_init", None) is not None
                has_hip = getattr(lib, "ggml_backend_hip_init", None) is not None or getattr(lib, "ggml_backend_rocm_init", None) is not None
                has_sycl = getattr(lib, "ggml_backend_sycl_init", None) is not None

                if has_vk or has_cuda or has_metal or has_hip or has_sycl:
                    gpu_offload_supported = True
                    if has_vk:
                        active_backend = "Vulkan (Evrensel GPU Hızlandırmalı)"
                    elif has_cuda:
                        active_backend = "CUDA (NVIDIA GPU Hızlandırmalı)"
                    elif has_metal:
                        active_backend = "Metal (Apple Silicon Hızlandırmalı)"
                    elif has_hip:
                        active_backend = "ROCm / HIP (AMD GPU Hızlandırmalı)"
                    elif has_sycl:
                        active_backend = "SYCL (Intel GPU Hızlandırmalı)"
                    else:
                        active_backend = "GPU Hızlandırmalı"
            
            if not gpu_offload_supported:
                cpu_info = HardwareChecker.get_cpu_info()
                active_backend = f"CPU (Standart - {cpu_info['optimal_threads']} İş Parçacığı)"
        except Exception:
            gpu_offload_supported = False
            active_backend = "CPU (Standart)"

        info = {
            "gpu_offload_supported": gpu_offload_supported,
            "active_backend": active_backend,
            "detected_gpus": gpus,
            "gpu_summary": gpu_str,
            "vulkan_installed": vulkan_installed,
            "cuda_installed": cuda_installed,
        }
        HardwareChecker._cached_acceleration_info = info
        return info

    @staticmethod
    def check_model_compatibility(model: LocalModelInfo) -> Tuple[bool, str, str]:
        """
        Checks if the system has enough memory to run the specified model.
        Blocking (is_runnable=False) is strictly based on TOTAL system RAM.
        Low available RAM produces a WARNING.
        Returns:
            (is_runnable: bool, message: str, status_level: 'OK' | 'WARNING' | 'ERROR')
        """
        mem = HardwareChecker.get_system_memory_info()
        available_gb = mem["available_gb"]
        total_gb = mem["total_gb"]

        # Only block if total physical RAM is strictly lower than minimum required RAM
        if total_gb < model.min_ram_gb:
            msg = (
                f"Yetersiz Donanım! Bu model en az {model.min_ram_gb:.0f} GB RAM gerektirir. "
                f"Sisteminizde toplam {total_gb:.1f} GB RAM bulunmaktadır. "
                "Bilgisayarınızın kilitlenmemesi için bu model engellendi."
            )
            return False, msg, "ERROR"

        # If available RAM is lower than min RAM, give warning but allow execution
        if available_gb < model.min_ram_gb:
            msg = (
                f"Düşük Boş Bellek Uyarısı: Model için ~{model.min_ram_gb:.0f} GB boş RAM önerilir, "
                f"şu an {available_gb:.1f} GB boş RAM var (Toplam RAM: {total_gb:.1f} GB). "
                "Model yüklenirken işletim sistemi takas (swap) belleği kullanabilir."
            )
            return True, msg, "WARNING"

        # If available RAM is between minimum and recommended
        if available_gb < model.recommended_ram_gb:
            msg = (
                f"Uygun (Önerilen Boş Bellek: {model.recommended_ram_gb:.0f} GB, "
                f"Şu an Boş: {available_gb:.1f} GB / Toplam: {total_gb:.1f} GB)."
            )
            return True, msg, "WARNING"

        # Fully compatible
        msg = f"Sistem tamamen uyumlu. (Toplam: {total_gb:.1f} GB, Boş: {available_gb:.1f} GB)"
        return True, msg, "OK"

    @staticmethod
    def check_file_compatibility(file_path: str) -> Tuple[bool, str, str]:
        """
        Checks compatibility for a custom GGUF file path based on its disk size.
        Blocking is based on TOTAL RAM. Low available RAM produces a WARNING.
        """
        if not os.path.exists(file_path):
            return False, "Model dosyası diskte bulunamadı.", "ERROR"

        size_bytes = os.path.getsize(file_path)
        size_gb = size_bytes / (1024 ** 3)
        min_ram_gb = (size_gb * 1.2) + 0.6
        rec_ram_gb = (size_gb * 1.5) + 1.0

        mem = HardwareChecker.get_system_memory_info()
        available_gb = mem["available_gb"]
        total_gb = mem["total_gb"]

        if total_gb < min_ram_gb:
            return False, (
                f"Yetersiz Donanım! Bu dosya (~{size_gb:.1f} GB) için en az {min_ram_gb:.1f} GB toplam RAM gereklidir. "
                f"Sistemdeki toplam RAM: {total_gb:.1f} GB."
            ), "ERROR"

        if available_gb < min_ram_gb:
            return True, (
                f"Düşük Boş Bellek Uyarısı: Gerekli ~{min_ram_gb:.1f} GB, mevcut boş RAM {available_gb:.1f} GB "
                f"(Toplam: {total_gb:.1f} GB)."
            ), "WARNING"

        if available_gb < rec_ram_gb:
            return True, f"Uygun. (Boş RAM: {available_gb:.1f} GB / Toplam: {total_gb:.1f} GB)", "WARNING"

        return True, f"Sistem uyumlu. (Boş RAM: {available_gb:.1f} GB / Toplam: {total_gb:.1f} GB)", "OK"

    @staticmethod
    def check_disk_space(target_dir: str, required_bytes: int) -> Tuple[bool, str, float]:
        """
        Checks if the disk containing target_dir has enough free space for required_bytes.
        Returns:
            (has_space: bool, message: str, free_gb: float)
        """
        try:
            os.makedirs(target_dir, exist_ok=True)
            usage = psutil.disk_usage(target_dir)
            free_bytes = usage.free
            free_gb = round(free_bytes / (1024 ** 3), 2)
            req_gb = round(required_bytes / (1024 ** 3), 2)

            # Require at least 5% safety margin or 500 MB
            safe_required = required_bytes + max(500 * 1024 * 1024, int(required_bytes * 0.05))

            if free_bytes < safe_required:
                msg = (
                    f"Yetersiz Disk Alanı! Bu model için diskte en az ~{req_gb:.1f} GB boş alan gereklidir. "
                    f"Seçili sürücüde yalnızca {free_gb:.1f} GB boş yer var."
                )
                return False, msg, free_gb

            return True, f"Disk alanı yeterli ({free_gb:.1f} GB boş).", free_gb
        except Exception as e:
            log_error(f"Error checking disk space: {e}")
            return True, "Disk alanı denetlenemedi, indirmeye izin veriliyor.", 0.0

    @staticmethod
    def get_device_performance_profile() -> str:
        """
        Classifies current host hardware profile:
        - 'LOW_END': Total RAM <= 8.5 GB, available RAM <= 3.5 GB, or no hardware acceleration
        - 'STANDARD': 12-16 GB RAM with discrete GPU or hardware acceleration
        - 'HIGH_END': 24+ GB RAM or 12+ GB VRAM
        """
        mem = HardwareChecker.get_system_memory_info()
        total_ram = mem["total_gb"]
        avail_ram = mem["available_gb"]
        accel = HardwareChecker.get_acceleration_info()
        has_gpu = accel.get("gpu_offload_supported", False)

        if total_ram <= 8.5 or avail_ram <= 3.5 or not has_gpu:
            return "LOW_END"
        if total_ram >= 24.0:
            return "HIGH_END"
        return "STANDARD"

    @staticmethod
    def calculate_adaptive_context_window(
        text_content: str = "",
        model_path: str = "",
        model_max_ctx: int = 131072
    ) -> int:
        """
        Calculates optimal, demand-driven context window:
        1. 'Don't strain system needlessly': Allocates only what document needs (+ headroom) for short texts.
        2. 'Low-End Safety': Limits context ceiling to 16K (or 8K) on constrained hardware.
        3. 'Granular Scaling': Expands up to model_max_ctx in 4K blocks when document and hardware demand it.
        Minimum context floor is 8192 tokens.
        """
        MIN_FLOOR = 8192

        # 1. Estimate document token requirement (~3 chars per token)
        doc_tokens = max(1, int(len(text_content) / 3.0)) if text_content else 0
        output_budget = 4096
        system_overhead = 2048
        needed_tokens = doc_tokens + output_budget + system_overhead

        # Round up to nearest 4096 block
        target_doc_ctx = int(((needed_tokens + 4095) // 4096) * 4096)
        target_doc_ctx = max(MIN_FLOOR, target_doc_ctx)

        # 2. Check hardware performance profile & memory limits
        profile = HardwareChecker.get_device_performance_profile()
        mem = HardwareChecker.get_system_memory_info()
        avail_ram_gb = mem["available_gb"]
        total_ram_gb = mem["total_gb"]

        # Safe ceilings by profile
        if profile == "LOW_END":
            # For weak / integrated / <=8GB RAM systems: never exceed 16K (or 8K if RAM < 2.5GB)
            max_safe_ceiling = 8192 if avail_ram_gb < 2.5 else 16384
        elif profile == "HIGH_END":
            max_safe_ceiling = min(model_max_ctx, 262144 if total_ram_gb >= 32.0 else 131072)
        else:  # STANDARD
            model_size_gb = (os.path.getsize(model_path) / (1024 ** 3)) if (model_path and os.path.exists(model_path)) else 3.5
            # For 12B+ models on 16GB RAM, keep max ceiling at 65536 to prevent VRAM overflow
            if model_size_gb > 5.5:
                max_safe_ceiling = min(model_max_ctx, 65536)
            else:
                max_safe_ceiling = min(model_max_ctx, 131072)

        # Final context is the minimum between what document demands and what hardware safely allows
        optimal_ctx = min(target_doc_ctx, max_safe_ceiling)
        optimal_ctx = max(MIN_FLOOR, optimal_ctx)

        log_debug(
            f"Adaptive context calculated: {optimal_ctx} tokens "
            f"(Profile: {profile}, DocDemand: {target_doc_ctx}, SafeCeiling: {max_safe_ceiling}, AvailRAM: {avail_ram_gb:.1f}GB)"
        )
        return optimal_ctx

