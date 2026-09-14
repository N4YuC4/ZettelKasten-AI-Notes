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
    "gemma-4-e2b": LocalModelInfo(
        id="gemma-4-e2b",
        display_name="Gemma 4 (E2B)",
        user_description="Hafif & Ultra Hızlı — 128K Doğal Bağlam, tüm bilgisayarlar için ideal",
        tier_label="Hafif",
        repo_id="mradermacher/Huihui-gemma-4-E2B-it-abliterated-i1-GGUF",
        filename="Huihui-gemma-4-E2B-it-abliterated.i1-Q4_K_M.gguf",
        size_bytes=3_427_874_560,
        min_ram_gb=4.0,
        recommended_ram_gb=8.0,
        context_window=131072,
    ),
    "gemma-4-12b": LocalModelInfo(
        id="gemma-4-12b",
        display_name="Gemma 4 (12B)",
        user_description="Dengeli & Derin Zeka — Karmaşık tezler ve çok katmanlı kavramsal analiz",
        tier_label="Dengeli",
        repo_id="mradermacher/Huihui-gemma-4-12B-it-abliterated-GGUF",
        filename="Huihui-gemma-4-12B-it-abliterated.IQ4_XS.gguf",
        size_bytes=6_690_000_000,
        min_ram_gb=8.0,
        recommended_ram_gb=16.0,
        context_window=131072,
    ),
    "gemma-4-26b-moe": LocalModelInfo(
        id="gemma-4-26b-moe",
        display_name="Gemma 4 (26B MoE)",
        user_description="Yüksek Kapasite — 26B Amiral gemisi uzman karması mimarisi",
        tier_label="Yüksek Kapasite",
        repo_id="mradermacher/gemma-4-26B-A4B-it-abliterated-GGUF",
        filename="gemma-4-26B-A4B-it-abliterated.IQ4_XS.gguf",
        size_bytes=14_060_000_000,
        min_ram_gb=16.0,
        recommended_ram_gb=32.0,
        context_window=131072,
    ),
}

DEFAULT_MODEL_ID = "gemma-4-e2b"


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

