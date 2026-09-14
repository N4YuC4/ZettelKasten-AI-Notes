# env_config.py
#
# Centralized process-level environment configuration for headless GPU compute and GUI stability.
# Prevents desktop presentation layer conflicts and deadlocks between Vulkan/EGL and Flutter/Flet.

import os


def configure_headless_environment() -> None:
    """
    Disables Vulkan loader desktop presentation layers and NVIDIA Optimus presentation hooks.
    Prevents mutex deadlocks and glibc priority protection assertions with Wayland/EGL/Flet.
    """
    os.environ["VK_LOADER_LAYERS_DISABLE"] = "*"
    os.environ["DISABLE_LAYER_NV_OPTIMUS_1"] = "1"
    os.environ["DISABLE_LAYER_NV_PRESENT_1"] = "1"

