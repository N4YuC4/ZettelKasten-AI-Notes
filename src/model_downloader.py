# model_downloader.py
#
# Background resumable model download manager for Hugging Face GGUF models.
# Supports automatic reconnection on TLS/SSL drops, resume from byte offsets,
# dynamic UI listener subscription, ETA calculation, and atomic file finalization.

import os
import time
import requests
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Any
from local_models_catalog import LocalModelInfo
from hardware_checker import HardwareChecker
from logger import log_debug, log_error


@dataclass
class DownloadStatus:
    """Represents the current real-time status of a model download."""
    model_id: str
    state: str = "idle"  # "idle" | "downloading" | "completed" | "error"
    progress: float = 0.0  # 0.0 to 1.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_mb_s: float = 0.0
    eta_seconds: Optional[int] = None
    status_text: str = ""
    error_message: Optional[str] = None
    retry_count: int = 0


def format_eta(seconds: Optional[int]) -> str:
    """Formats seconds into human-readable ETA string."""
    if seconds is None or seconds <= 0:
        return ""
    if seconds < 60:
        return f" • Remaining: {seconds}s"
    elif seconds < 3600:
        m = seconds // 60
        s = seconds % 60
        return f" • Remaining: {m}m {s}s" if s > 0 else f" • Remaining: {m}m"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f" • Remaining: {h}h {m}m"


class ModelDownloader:
    """
    Manages asynchronous, resumable downloads of GGUF model files
    from Hugging Face CDN to local storage with progress reporting.
    Features state registry and dynamic listener subscription for seamless UI binding.
    """
    _active_threads: Dict[str, threading.Thread] = {}
    _cancel_events: Dict[str, threading.Event] = {}
    _status_registry: Dict[str, DownloadStatus] = {}
    _listeners: Dict[str, List[Callable[[DownloadStatus], None]]] = {}
    _lock = threading.Lock()

    @classmethod
    def get_model_path(cls, model: LocalModelInfo, models_dir: str) -> str:
        """Returns the full local destination path for a given model."""
        return os.path.join(models_dir, model.filename)

    @classmethod
    def is_model_downloaded(cls, model: LocalModelInfo, models_dir: str) -> bool:
        """Checks if a model file exists locally and has non-zero size."""
        target_path = cls.get_model_path(model, models_dir)
        return os.path.isfile(target_path) and os.path.getsize(target_path) > 10_000_000

    @classmethod
    def delete_model(cls, model: LocalModelInfo, models_dir: str) -> bool:
        """Deletes a downloaded model file and any associated partial files."""
        target_path = cls.get_model_path(model, models_dir)
        part_path = target_path + ".part"
        deleted = False
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                deleted = True
                log_debug(f"Deleted model file: {target_path}")
            except Exception as e:
                log_error(f"Failed to delete model {target_path}: {e}")
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
                deleted = True
            except Exception:
                pass

        with cls._lock:
            cls._status_registry[model.id] = DownloadStatus(model_id=model.id, state="idle")
        cls._notify_listeners(model.id)
        return deleted

    @classmethod
    def is_downloading(cls, model_id: str) -> bool:
        """Returns True if an active download thread is running for the model."""
        thread = cls._active_threads.get(model_id)
        return thread is not None and thread.is_alive()

    @classmethod
    def get_status(cls, model_id: str) -> DownloadStatus:
        """Retrieves or initializes the current download status for a model."""
        with cls._lock:
            if model_id not in cls._status_registry:
                cls._status_registry[model_id] = DownloadStatus(model_id=model_id, state="idle")
            return cls._status_registry[model_id]

    @classmethod
    def register_listener(cls, model_id: str, listener: Callable[[DownloadStatus], None]) -> None:
        """Subscribes a UI callback to real-time status updates for a model."""
        with cls._lock:
            if model_id not in cls._listeners:
                cls._listeners[model_id] = []
            if listener not in cls._listeners[model_id]:
                cls._listeners[model_id].append(listener)
        # Emit current status immediately on registration
        current_status = cls.get_status(model_id)
        try:
            listener(current_status)
        except Exception:
            pass

    @classmethod
    def unregister_listener(cls, model_id: str, listener: Callable[[DownloadStatus], None]) -> None:
        """Unsubscribes a UI callback from status updates."""
        with cls._lock:
            if model_id in cls._listeners and listener in cls._listeners[model_id]:
                cls._listeners[model_id].remove(listener)

    @classmethod
    def _notify_listeners(cls, model_id: str) -> None:
        """Dispatches the current status to all subscribed listeners."""
        with cls._lock:
            status = cls._status_registry.get(model_id)
            listeners = list(cls._listeners.get(model_id, []))
        if status:
            for listener in listeners:
                try:
                    listener(status)
                except Exception as ex:
                    log_error(f"Error in download listener: {ex}")

    @classmethod
    def cancel_download(cls, model_id: str) -> None:
        """Signals an ongoing download to cancel gracefully."""
        event = cls._cancel_events.get(model_id)
        if event:
            event.set()
            log_debug(f"Cancel signal sent for download: {model_id}")
        with cls._lock:
            if model_id in cls._status_registry:
                st = cls._status_registry[model_id]
                st.state = "idle"
                st.status_text = "Download cancelled."
        cls._notify_listeners(model_id)

    @classmethod
    def start_download(
        cls,
        model: LocalModelInfo,
        models_dir: str,
        on_progress: Optional[Callable[[float, str, float], None]] = None,
        on_finished: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Starts an asynchronous, resumable download with automatic reconnection.
        """
        if cls.is_downloading(model.id):
            log_debug(f"Download already in progress for {model.id}")
            return

        # Pre-check disk space
        has_space, space_msg, _ = HardwareChecker.check_disk_space(models_dir, model.size_bytes)
        if not has_space:
            log_error(f"Cannot start download for {model.id}: {space_msg}")
            with cls._lock:
                cls._status_registry[model.id] = DownloadStatus(
                    model_id=model.id,
                    state="error",
                    status_text=space_msg,
                    error_message=space_msg
                )
            cls._notify_listeners(model.id)
            if on_error:
                on_error(space_msg)
            return

        cancel_event = threading.Event()
        cls._cancel_events[model.id] = cancel_event

        with cls._lock:
            cls._status_registry[model.id] = DownloadStatus(
                model_id=model.id,
                state="downloading",
                progress=0.0,
                total_bytes=model.size_bytes,
                status_text="Connecting..."
            )
        cls._notify_listeners(model.id)

        # Register direct callbacks if provided
        if on_progress:
            def _prog_adapter(st: DownloadStatus):
                if st.state in ("downloading", "completed"):
                    on_progress(st.progress, st.status_text, st.speed_mb_s)
            cls.register_listener(model.id, _prog_adapter)

        def _worker():
            os.makedirs(models_dir, exist_ok=True)
            dest_path = cls.get_model_path(model, models_dir)
            part_path = dest_path + ".part"

            start_time = time.time()
            bytes_since_start = 0
            last_update_time = 0
            total_bytes = model.size_bytes

            MAX_RETRIES = 30
            retry_count = 0

            while retry_count < MAX_RETRIES:
                if cancel_event.is_set():
                    log_debug(f"Download cancelled for {model.id}")
                    return

                downloaded_bytes = os.path.getsize(part_path) if os.path.exists(part_path) else 0

                # If file is already fully downloaded
                if total_bytes and downloaded_bytes >= total_bytes:
                    break

                headers = {
                    "User-Agent": "ZettelKasten-AI-Notes/1.0",
                }
                if downloaded_bytes > 0:
                    headers["Range"] = f"bytes={downloaded_bytes}-"

                session = requests.Session()
                bytes_in_attempt = 0
                try:
                    log_debug(
                        f"Starting download: {model.download_url} "
                        f"(resuming from {downloaded_bytes} bytes, attempt {retry_count + 1})"
                    )
                    response = session.get(
                        model.download_url,
                        headers=headers,
                        stream=True,
                        timeout=(15, 60),
                        allow_redirects=True
                    )

                    if response.status_code == 416:
                        # Range Not Satisfiable -> Already complete
                        session.close()
                        break
                    elif response.status_code not in (200, 206):
                        session.close()
                        raise RuntimeError(f"HTTP {response.status_code}: {response.reason}")

                    # Parse exact total size from Content-Range or Content-Length
                    content_range = response.headers.get("content-range")
                    if content_range and "/" in content_range:
                        total_part = content_range.split("/")[-1].strip()
                        if total_part.isdigit():
                            total_bytes = int(total_part)
                    elif response.status_code == 206:
                        content_length = int(response.headers.get("content-length", 0))
                        total_bytes = downloaded_bytes + content_length
                    else:
                        content_length = int(response.headers.get("content-length", 0))
                        total_bytes = content_length or model.size_bytes
                        downloaded_bytes = 0

                    mode = "ab" if (downloaded_bytes > 0 and response.status_code == 206) else "wb"
                    chunk_size = 1024 * 256  # 256 KB chunks for stable TLS socket

                    with open(part_path, mode) as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if cancel_event.is_set():
                                log_debug(f"Download cancelled for {model.id}")
                                session.close()
                                return

                            if chunk:
                                f.write(chunk)
                                downloaded_bytes += len(chunk)
                                bytes_since_start += len(chunk)
                                bytes_in_attempt += len(chunk)

                                # Reset consecutive failure counter once active data transfer is confirmed
                                if retry_count > 0:
                                    log_debug(
                                        f"Active download confirmed for {model.id} "
                                        f"({bytes_in_attempt} bytes received in attempt). "
                                        f"Resetting consecutive retry count from {retry_count} to 0."
                                    )
                                    retry_count = 0

                                now = time.time()
                                if now - last_update_time >= 0.2:  # Update 5 times/sec
                                    elapsed = max(0.001, now - start_time)
                                    speed_mb_s = (bytes_since_start / (1024 * 1024)) / elapsed
                                    progress = min(1.0, max(0.0, downloaded_bytes / total_bytes)) if total_bytes > 0 else 0.0

                                    cur_gb = downloaded_bytes / (1024 ** 3)
                                    tot_gb = total_bytes / (1024 ** 3)

                                    remaining_bytes = max(0, total_bytes - downloaded_bytes)
                                    eta_seconds = int(remaining_bytes / (speed_mb_s * 1024 * 1024)) if speed_mb_s > 0.05 else None
                                    eta_str = format_eta(eta_seconds)

                                    status_text = f"{cur_gb:.2f} GB / {tot_gb:.2f} GB ({int(progress * 100)}%) • {speed_mb_s:.1f} MB/s{eta_str}"

                                    with cls._lock:
                                        cls._status_registry[model.id] = DownloadStatus(
                                            model_id=model.id,
                                            state="downloading",
                                            progress=progress,
                                            downloaded_bytes=downloaded_bytes,
                                            total_bytes=total_bytes,
                                            speed_mb_s=speed_mb_s,
                                            eta_seconds=eta_seconds,
                                            status_text=status_text,
                                            retry_count=retry_count
                                        )
                                    cls._notify_listeners(model.id)
                                    last_update_time = now

                    session.close()

                    # Check completion
                    if (total_bytes and downloaded_bytes >= total_bytes) or (response.status_code == 200 and downloaded_bytes > 0):
                        break

                except Exception as e:
                    session.close()
                    if cancel_event.is_set():
                        return
                    retry_count += 1
                    err_msg = str(e)
                    log_error(f"Download error for {model.id} (Attempt {retry_count}/{MAX_RETRIES}): {err_msg}")

                    cur_downloaded = os.path.getsize(part_path) if os.path.exists(part_path) else 0
                    cur_gb = cur_downloaded / (1024 ** 3)
                    tot_gb = total_bytes / (1024 ** 3) if total_bytes else model.size_gb
                    progress = min(1.0, max(0.0, cur_downloaded / total_bytes)) if total_bytes else 0.0

                    retry_status = f"{cur_gb:.2f} GB / {tot_gb:.2f} GB — Connection dropped, automatically reconnecting... ({retry_count}/{MAX_RETRIES})"

                    with cls._lock:
                        cls._status_registry[model.id] = DownloadStatus(
                            model_id=model.id,
                            state="downloading",
                            progress=progress,
                            downloaded_bytes=cur_downloaded,
                            total_bytes=total_bytes,
                            speed_mb_s=0.0,
                            eta_seconds=None,
                            status_text=retry_status,
                            retry_count=retry_count
                        )
                    cls._notify_listeners(model.id)

                    if retry_count >= MAX_RETRIES:
                        final_err = f"Download failed ({MAX_RETRIES} consecutive attempts): {err_msg}"
                        with cls._lock:
                            cls._status_registry[model.id] = DownloadStatus(
                                model_id=model.id,
                                state="error",
                                progress=progress,
                                downloaded_bytes=cur_downloaded,
                                total_bytes=total_bytes,
                                status_text=final_err,
                                error_message=final_err,
                                retry_count=retry_count
                            )
                        cls._notify_listeners(model.id)
                        if on_error:
                            try:
                                on_error(final_err)
                            except Exception:
                                pass
                        return

                    time.sleep(min(5, 1.5 * retry_count))

            # Finalize: rename .part to final target
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.rename(part_path, dest_path)
                log_debug(f"Download complete: {dest_path}")

                with cls._lock:
                    cls._status_registry[model.id] = DownloadStatus(
                        model_id=model.id,
                        state="completed",
                        progress=1.0,
                        downloaded_bytes=total_bytes,
                        total_bytes=total_bytes,
                        status_text="Download complete!"
                    )
                cls._notify_listeners(model.id)

                if on_finished:
                    try:
                        on_finished(dest_path)
                    except Exception as fe:
                        log_error(f"Error in on_finished callback: {fe}")
            except Exception as fin_err:
                log_error(f"Failed to finalize downloaded file: {fin_err}")
                err_text = f"Failed to save file: {fin_err}"
                with cls._lock:
                    cls._status_registry[model.id] = DownloadStatus(
                        model_id=model.id,
                        state="error",
                        status_text=err_text,
                        error_message=err_text
                    )
                cls._notify_listeners(model.id)
                if on_error:
                    try:
                        on_error(err_text)
                    except Exception:
                        pass
            finally:
                cls._active_threads.pop(model.id, None)
                cls._cancel_events.pop(model.id, None)

        thread = threading.Thread(target=_worker, daemon=True)
        cls._active_threads[model.id] = thread
        thread.start()
