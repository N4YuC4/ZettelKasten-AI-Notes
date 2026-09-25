# reranker_service.py
#
# Dedicated cross-encoder reranking service for Zettelkasten AI Notes.
# Powers stage-2 candidate re-scoring using Qwen3-Reranker-0.6B-GGUF to eliminate
# duplicate concepts, distinguish fine-grained nuances, and refine knowledge retrieval.
# Runs strictly on CPU (n_threads=2) to isolate compute and eliminate VRAM conflicts
# with GPU generation LLMs.

import os
import threading
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from logger import log_debug, log_error
import local_models_catalog


class RerankerModelNotFoundError(FileNotFoundError):
    """Raised when the mandatory Qwen3-Reranker GGUF model file does not exist on disk."""
    pass


class RerankerInferenceError(RuntimeError):
    """Raised when local reranker cross-encoder inference fails."""
    pass


class RerankerService:
    """
    Manages local cross-encoder scoring for (query, candidate_document) pairs.
    Powered by Qwen3-Reranker-0.6B (GGUF format, instruction-aware cross-attention).
    Strictly executed on CPU to preserve 100% of GPU VRAM for generation LLMs.
    """

    _cached_llm: Optional[Any] = None
    _cached_model_path: Optional[str] = None
    _tok_yes: Optional[int] = None
    _tok_no: Optional[int] = None
    _lock = threading.Lock()

    DEFAULT_INSTRUCTION = (
        "Given a document chunk, evaluate whether the existing note covers the same "
        "theoretical mechanism or closely related concept."
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_threads: Optional[int] = None,
        n_ctx: int = 4096,
    ):
        if model_path:
            self.model_path = os.path.abspath(model_path)
        else:
            default_dir = local_models_catalog.get_default_models_dir()
            qwen_info = local_models_catalog.get_model_by_id(
                local_models_catalog.DEFAULT_RERANKER_MODEL_ID
            )
            filename = qwen_info.filename if qwen_info else "Qwen3-Reranker-0.6B.Q4_K_M.gguf"
            self.model_path = os.path.abspath(os.path.join(default_dir, filename))

        self.n_threads = n_threads or 2
        self.n_ctx = n_ctx

    def is_model_available(self) -> bool:
        """Checks if the local reranker model file exists on disk."""
        return os.path.exists(self.model_path)

    def _get_or_load_reranker(self) -> Any:
        """Retrieves cached Llama instance configured for cross-encoder ranking or initializes a new one on CPU."""
        if not self.is_model_available():
            raise RerankerModelNotFoundError(
                f"Reranker model is mandatory but not found at: '{self.model_path}'. "
                "Please download Qwen3 Reranker (0.6B) via Settings -> Model Manager before generating notes."
            )

        with RerankerService._lock:
            if (
                RerankerService._cached_llm is not None
                and RerankerService._cached_model_path == self.model_path
            ):
                return RerankerService._cached_llm

            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise RerankerInferenceError(
                    "The llama-cpp-python library is required for local reranking. "
                    "Please run 'pip install llama-cpp-python'."
                ) from e

            log_debug(
                f"Loading local reranker model: {self.model_path} "
                f"(CPU-only, threads={self.n_threads}, n_ctx={self.n_ctx})"
            )

            old_vk = os.environ.get("VK_DRIVER_FILES")
            try:
                os.environ["VK_DRIVER_FILES"] = ""
                reranker = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=0,  # Strictly CPU to prevent VRAM competition with generator LLM
                    n_threads=self.n_threads,
                    n_ctx=self.n_ctx,
                    verbose=False
                )
                RerankerService._cached_llm = reranker
                RerankerService._cached_model_path = self.model_path

                # Resolve and cache token IDs for yes/no
                try:
                    tok_yes_list = reranker.tokenize(b"yes")
                    tok_no_list = reranker.tokenize(b"no")
                    RerankerService._tok_yes = tok_yes_list[-1] if tok_yes_list else 9693
                    RerankerService._tok_no = tok_no_list[-1] if tok_no_list else 2152
                except Exception as te:
                    log_debug(f"Could not tokenize yes/no tokens from reranker ({te}), falling back to default IDs.")
                    RerankerService._tok_yes = 9693
                    RerankerService._tok_no = 2152

                log_debug(
                    f"Local reranker model loaded successfully into CPU memory "
                    f"(tok_yes={RerankerService._tok_yes}, tok_no={RerankerService._tok_no})."
                )
                return reranker
            except Exception as e:
                log_error(f"Failed to initialize local reranker model: {e}")
                raise RerankerInferenceError(f"Could not load reranker model at {self.model_path}: {e}") from e
            finally:
                if old_vk is not None:
                    os.environ["VK_DRIVER_FILES"] = old_vk
                else:
                    os.environ.pop("VK_DRIVER_FILES", None)

    @classmethod
    def unload_cached_model(cls) -> None:
        """Frees the cached reranker model and context from CPU memory."""
        with cls._lock:
            if cls._cached_llm is not None:
                log_debug(f"Unloading cached reranker model: {cls._cached_model_path}")
                try:
                    if hasattr(cls._cached_llm, "close"):
                        cls._cached_llm.close()
                except Exception as e:
                    log_error(f"Error while closing reranker model: {e}")
                finally:
                    cls._cached_llm = None
                    cls._cached_model_path = None
                    cls._tok_yes = None
                    cls._tok_no = None
                    log_debug("Cached reranker model memory freed.")

    @staticmethod
    def format_rerank_input(
        query: str,
        title: str,
        content: str,
        instruction: Optional[str] = None
    ) -> str:
        """
        Formats input according to official Qwen3-Reranker instruction-aware template:
        <|im_start|>system
        {instruction}<|im_end|>
        <|im_start|>user
        <Query>: {query}
        <Document>: Title: {title}
        {content}<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        """
        inst = instruction or RerankerService.DEFAULT_INSTRUCTION
        # Truncate content to safe character bounds to prevent context overrun
        q_clean = str(query).strip()[:2000]
        t_clean = str(title).strip()
        c_clean = str(content).strip()[:2000]
        return (
            f"<|im_start|>system\n{inst}<|im_end|>\n"
            f"<|im_start|>user\n<Query>: {q_clean}\n<Document>: Title: {t_clean}\n{c_clean}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid function mapping logits to [0.0, 1.0]."""
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def score_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        instruction: Optional[str] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Scores candidate notes against the target query using native Qwen3 logit difference:
        score = sigmoid(logit("yes") - logit("no"))
        Returns a list of (score, candidate_dict) pairs sorted descending by score.
        Raises RerankerModelNotFoundError if model is not available.
        """
        if not candidates:
            return []

        reranker = self._get_or_load_reranker()

        tok_yes = getattr(RerankerService, "_tok_yes", None)
        tok_no = getattr(RerankerService, "_tok_no", None)
        if tok_yes is None or tok_no is None:
            try:
                tok_yes_list = reranker.tokenize(b"yes")
                tok_no_list = reranker.tokenize(b"no")
                tok_yes = tok_yes_list[-1] if tok_yes_list else 9693
                tok_no = tok_no_list[-1] if tok_no_list else 2152
            except Exception:
                tok_yes = 9693
                tok_no = 2152
            RerankerService._tok_yes = tok_yes
            RerankerService._tok_no = tok_no

        scored_results: List[Tuple[float, Dict[str, Any]]] = []

        for candidate in candidates:
            prompt = self.format_rerank_input(
                query=query,
                title=candidate.get("title", ""),
                content=candidate.get("content", ""),
                instruction=instruction
            )

            try:
                tokens = reranker.tokenize(prompt.encode("utf-8"))
                # Clamp tokens to safe context window budget
                max_allowed = max(64, self.n_ctx - 10)
                if len(tokens) > max_allowed:
                    tokens = tokens[:max_allowed]

                reranker.reset()
                reranker.eval(tokens)

                # Extract logits at the last position (-1)
                logits_ptr = reranker._ctx.get_logits_ith(-1)
                logit_yes = float(logits_ptr[tok_yes])
                logit_no = float(logits_ptr[tok_no])

                logit_diff = logit_yes - logit_no
                norm_score = self._sigmoid(logit_diff)
                scored_results.append((norm_score, candidate))
            except Exception as e:
                log_error(f"Reranker inference error for candidate '{candidate.get('title')}': {e}")
                raise RerankerInferenceError(f"Cross-encoder scoring failed: {e}") from e

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.0,
        instruction: Optional[str] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Reranks candidates and returns the top_k highest-scoring notes exceeding min_score.
        """
        scored = self.score_candidates(query, candidates, instruction=instruction)
        filtered = [item for item in scored if item[0] >= min_score]
        return filtered[:top_k]

