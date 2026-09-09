# local_models_catalog.py
#
# Curated catalog of tested and verified local GGUF models.
# Provides clean user-facing names, simplified descriptions, and direct download links.

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LocalModelInfo:
    """Represents metadata for a curated local GGUF model."""
    id: str
    display_name: str
    user_description: str
    tier_label: str
    repo_id: str
    filename: str
    size_bytes: int
    min_ram_gb: float
    recommended_ram_gb: float
    context_window: int = 32768

    @property
    def download_url(self) -> str:
        """Returns the direct download URL from Hugging Face CDN."""
        return f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"

    @property
    def size_gb(self) -> float:
        """Returns the approximate file size in gigabytes."""
        return self.size_bytes / (1024 ** 3)


# Curated list of tested local models with clean, end-user friendly names and descriptions
CURATED_MODELS: Dict[str, LocalModelInfo] = {
    "qwen-3.5-4b": LocalModelInfo(
        id="qwen-3.5-4b",
        display_name="Qwen 3.5 (4B)",
        user_description="Hafif & Hızlı — Standart bilgisayarlar için ideal",
        tier_label="Hafif",
        repo_id="mradermacher/Qwen3.5-4B_Abliterated-GGUF",
        filename="Qwen3.5-4B_Abliterated.Q4_K_M.gguf",
        size_bytes=2_480_000_000,
        min_ram_gb=4.0,
        recommended_ram_gb=8.0,
    ),
    "qwen-3.5-9b": LocalModelInfo(
        id="qwen-3.5-9b",
        display_name="Qwen 3.5 (9B)",
        user_description="Dengeli & Detaylı — Yüksek doğruluk ve akıcı notlar",
        tier_label="Dengeli",
        repo_id="mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF",
        filename="Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf",
        size_bytes=5_530_000_000,
        min_ram_gb=8.0,
        recommended_ram_gb=16.0,
    ),
    "qwen-3.6-35b-moe": LocalModelInfo(
        id="qwen-3.6-35b-moe",
        display_name="Qwen 3.6 (35B MoE)",
        user_description="Gelişmiş Zeka & Yüksek Hız — Derin kavramsal ilişkiler",
        tier_label="Yüksek Kapasite",
        repo_id="huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF",
        filename="Huihui-Qwen3.6-35B-A3B-abliterated-MTP.Q4_K_M.gguf",
        size_bytes=21_500_000_000,
        min_ram_gb=24.0,
        recommended_ram_gb=32.0,
    ),
}

DEFAULT_MODEL_ID = "qwen-3.5-4b"


def get_curated_models() -> List[LocalModelInfo]:
    """Returns a list of all curated models in order of tier."""
    return list(CURATED_MODELS.values())


def get_model_by_id(model_id: str) -> Optional[LocalModelInfo]:
    """Retrieves model metadata by its ID."""
    return CURATED_MODELS.get(model_id)


def get_default_models_dir() -> str:
    """Returns the default directory for storing downloaded models."""
    # Store under user data dir or app root 'models'
    base_dir = os.path.join(os.path.expanduser("~"), ".local", "share", "zettelkasten_ai", "models")
    return base_dir

