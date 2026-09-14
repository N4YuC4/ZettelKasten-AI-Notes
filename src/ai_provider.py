# ai_provider.py
#
# Abstract base class and factory for AI note generation providers.
# Adheres to SOLID principles: Dependency Inversion (DIP) and Open/Closed (OCP).

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable


class BaseAiProvider(ABC):
    """
    Abstract interface for AI note generation providers (Cloud, Local, etc.).
    Adheres to SOLID Dependency Inversion (DIP) and Open/Closed (OCP) principles.
    """

    @abstractmethod
    def generate_zettelkasten_notes(
        self,
        text_content: str,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts structured Zettelkasten notes from raw text.
        Returns a list of dicts with keys: 'general_title', 'title', 'content', 'connections'.
        """
        pass

    def generate_note_links(
        self,
        notes: List[Dict[str, Any]],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Resolves or enriches inter-note semantic graph links.
        Default implementation returns notes unmodified for providers that generate links in a single pass.
        """
        return notes


def create_ai_provider(
    provider_type: str,
    db_manager: Optional[Any] = None,
    **kwargs: Any
) -> BaseAiProvider:
    """
    Factory function to instantiate the appropriate BaseAiProvider implementation.
    """
    p_type = (provider_type or "gemini").lower().strip()
    if p_type == "local":
        from local_gguf_client import LocalGgufClient
        import local_models_catalog
        import model_downloader

        model_path = kwargs.get("model_path")
        n_gpu_layers = kwargs.get("n_gpu_layers")

        if not model_path and db_manager:
            active_model_id = db_manager.get_setting("ACTIVE_LOCAL_MODEL") or local_models_catalog.DEFAULT_MODEL_ID
            models_dir = db_manager.get_setting("MODELS_DIR") or local_models_catalog.get_default_models_dir()
            model_info = local_models_catalog.get_model_by_id(active_model_id)
            if not model_info:
                from local_gguf_client import LocalModelNotFoundError
                raise LocalModelNotFoundError(f"Selected model not found in catalog: '{active_model_id}'")
            model_path = model_downloader.ModelDownloader.get_model_path(model_info, models_dir)
            if not model_downloader.ModelDownloader.is_model_downloaded(model_info, models_dir):
                from local_gguf_client import LocalModelNotFoundError
                raise LocalModelNotFoundError(
                    f"'{model_info.display_name}' has not been downloaded yet. "
                    "Please download the model via Settings -> Model Manager."
                )

        if n_gpu_layers is None and db_manager:
            gpu_accel = (db_manager.get_setting("GPU_ACCELERATION") != "False")
            n_gpu_layers = -1 if gpu_accel else 0
        elif n_gpu_layers is None:
            n_gpu_layers = -1

        return LocalGgufClient(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_threads=kwargs.get("n_threads"),
            text_content=kwargs.get("text_content")
        )
    else:
        from gemini_api_client import GeminiApiClient
        api_key = kwargs.get("api_key")
        return GeminiApiClient(api_key=api_key)
