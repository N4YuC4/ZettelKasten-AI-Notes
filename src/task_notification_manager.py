# task_notification_manager.py
#
# Dedicated task and notification state manager coordinating background tasks,
# real-time progress tracking, sliding corner toast notifications, and persistent notification history.

import uuid
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from logger import log_debug, log_error


@dataclass
class NotificationItem:
    """Represents a toast / persistent notification event."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    message: str = ""
    notification_type: str = "info"  # "info" | "success" | "warning" | "error"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    duration: float = 4.0            # Duration in seconds for corner toast display
    read: bool = False


@dataclass
class TaskItem:
    """Represents an ongoing or completed background task."""
    id: str
    title: str
    status_text: str = ""
    progress: Optional[float] = None  # 0.0 to 1.0, or None for indeterminate
    is_running: bool = True
    cancellable: bool = True
    on_cancel: Optional[Callable[[], None]] = None
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    finished_at: Optional[str] = None
    error: Optional[str] = None


class TaskNotificationManager:
    """
    Thread-safe manager for background tasks and notifications.
    Supports non-blocking background workers, live progress reporting,
    sliding toast notification dispatching, and notification history.
    """

    def __init__(self, max_history: int = 50):
        self._max_history = max_history
        self._tasks: Dict[str, TaskItem] = {}
        self._notifications: List[NotificationItem] = []
        self._listeners: List[Callable[[], None]] = []
        self._toast_listeners: List[Callable[[NotificationItem], None]] = []
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # Task Lifecycle Management
    # -------------------------------------------------------------------------

    def start_task(
        self,
        task_id: str,
        title: str,
        initial_status: str = "",
        on_cancel: Optional[Callable[[], None]] = None,
        cancellable: bool = True,
        progress: Optional[float] = None,
    ) -> TaskItem:
        """Registers and starts a new ongoing background task."""
        with self._lock:
            task = TaskItem(
                id=task_id,
                title=title,
                status_text=initial_status,
                progress=progress,
                is_running=True,
                cancellable=cancellable,
                on_cancel=on_cancel,
            )
            self._tasks[task_id] = task
            log_debug(f"[TaskManager] Started task '{task_id}': {title}")
        self._notify_listeners()
        return task

    def update_task(
        self,
        task_id: str,
        status_text: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> None:
        """Updates the status message and/or progress of an active task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if status_text is not None:
                task.status_text = status_text
            if progress is not None:
                task.progress = max(0.0, min(1.0, progress))
        self._notify_listeners()

    def finish_task(
        self,
        task_id: str,
        completion_message: Optional[str] = None,
        notification_title: Optional[str] = None,
        show_toast: bool = True,
    ) -> None:
        """Marks a task as finished and optionally issues a success toast."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.is_running = False
            task.progress = 1.0
            task.finished_at = datetime.now().strftime("%H:%M:%S")
            if completion_message:
                task.status_text = completion_message
            log_debug(f"[TaskManager] Finished task '{task_id}'")

        self._notify_listeners()

        if show_toast:
            title = notification_title or "Task Completed"
            msg = completion_message or f"'{task.title}' completed successfully."
            self.add_notification(
                title=title,
                message=msg,
                notification_type="success",
                show_toast=True,
            )

    def fail_task(
        self,
        task_id: str,
        error_message: str,
        notification_title: Optional[str] = None,
        show_toast: bool = True,
    ) -> None:
        """Marks a task as failed with an error message and issues an error toast."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.is_running = False
            task.error = error_message
            task.status_text = f"Error: {error_message}"
            task.finished_at = datetime.now().strftime("%H:%M:%S")
            log_error(f"[TaskManager] Task '{task_id}' failed: {error_message}")

        self._notify_listeners()

        if show_toast:
            title = notification_title or "Task Error"
            self.add_notification(
                title=title,
                message=error_message,
                notification_type="error",
                show_toast=True,
                duration=5.0,
            )

    def cancel_task(
        self,
        task_id: str,
        execute_callback: bool = True,
        show_toast: bool = True,
    ) -> None:
        """
        Cancels an active task cleanly.
        Guarded against re-entrant calls and duplicate notifications.
        """
        callback = None
        task_title = ""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or not task.is_running:
                # Task does not exist or was already cancelled; prevent duplicate execution
                return
            callback = task.on_cancel if execute_callback else None
            task.on_cancel = None  # Clear to prevent re-entrant recursion
            task_title = task.title
            task.is_running = False
            task.status_text = "Operation cancelled by user."
            task.finished_at = datetime.now().strftime("%H:%M:%S")
            log_debug(f"[TaskManager] Task '{task_id}' cancelled by user.")

        self._notify_listeners()

        initial_notif_count = len(self._notifications)

        if callback:
            try:
                callback()
            except Exception as e:
                log_error(f"Error executing cancel callback for task '{task_id}': {e}")

        # If show_toast is True AND callback didn't already dispatch a notification (e.g. via show_snack_bar)
        if show_toast and len(self._notifications) == initial_notif_count:
            self.add_notification(
                title="Task Cancelled",
                message=f"'{task_title}' operation stopped.",
                notification_type="warning",
                show_toast=True,
            )

    def remove_task(self, task_id: str) -> None:
        """Removes a completed or cancelled task from the active tasks registry."""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
        self._notify_listeners()

    def get_task(self, task_id: str) -> Optional[TaskItem]:
        """Retrieves a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_active_tasks(self) -> List[TaskItem]:
        """Returns all currently running tasks."""
        with self._lock:
            return [t for t in self._tasks.values() if t.is_running]

    def get_all_tasks(self) -> List[TaskItem]:
        """Returns all registered tasks (active and recently finished)."""
        with self._lock:
            return list(self._tasks.values())

    def get_active_task_count(self) -> int:
        """Returns number of active running tasks."""
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.is_running)

    # -------------------------------------------------------------------------
    # Notification & Toast Management
    # -------------------------------------------------------------------------

    def add_notification(
        self,
        title: str,
        message: str,
        notification_type: str = "info",
        show_toast: bool = True,
        duration: float = 4.0,
    ) -> NotificationItem:
        """
        Adds a notification to persistent history and optionally triggers a corner toast.
        """
        notif = NotificationItem(
            title=title,
            message=message,
            notification_type=notification_type,
            duration=duration,
        )
        with self._lock:
            self._notifications.insert(0, notif)
            if len(self._notifications) > self._max_history:
                self._notifications = self._notifications[: self._max_history]

        self._notify_listeners()

        if show_toast:
            self._notify_toast(notif)

        return notif

    def show_toast(
        self,
        title: str,
        message: str,
        notification_type: str = "info",
        duration: float = 4.0,
    ) -> NotificationItem:
        """Convenience method to dispatch a notification and corner toast."""
        return self.add_notification(
            title=title,
            message=message,
            notification_type=notification_type,
            show_toast=True,
            duration=duration,
        )

    def get_notifications(self) -> List[NotificationItem]:
        """Returns copy of the notification history list."""
        with self._lock:
            return list(self._notifications)

    def clear_notifications(self) -> None:
        """Clears all historical notifications."""
        with self._lock:
            self._notifications.clear()
        self._notify_listeners()

    def remove_notification(self, notif_id: str) -> None:
        """Removes a single notification from history by its ID."""
        with self._lock:
            self._notifications = [n for n in self._notifications if n.id != notif_id]
        self._notify_listeners()

    # -------------------------------------------------------------------------
    # Listener Subscription
    # -------------------------------------------------------------------------

    def subscribe(self, listener: Callable[[], None]) -> None:
        """Subscribes a listener to general state changes (tasks/history)."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[], None]) -> None:
        """Unsubscribes a listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def subscribe_toast(self, listener: Callable[[NotificationItem], None]) -> None:
        """Subscribes a listener specifically for incoming toast triggers."""
        with self._lock:
            if listener not in self._toast_listeners:
                self._toast_listeners.append(listener)

    def unsubscribe_toast(self, listener: Callable[[NotificationItem], None]) -> None:
        """Unsubscribes a toast listener."""
        with self._lock:
            if listener in self._toast_listeners:
                self._toast_listeners.remove(listener)

    def _notify_listeners(self) -> None:
        """Invokes all general state change listeners."""
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener()
            except Exception as e:
                log_error(f"Error in TaskNotificationManager listener: {e}")

    def _notify_toast(self, notif: NotificationItem) -> None:
        """Dispatches a toast event to all registered toast listeners."""
        with self._lock:
            toast_listeners = list(self._toast_listeners)
        for listener in toast_listeners:
            try:
                listener(notif)
            except Exception as e:
                log_error(f"Error in TaskNotificationManager toast listener: {e}")

    def bind_model_downloader(self) -> None:
        """Subscribes to ModelDownloader events for all catalog models to track background downloads."""
        try:
            import local_models_catalog
            from model_downloader import ModelDownloader, format_eta

            for model in local_models_catalog.get_all_models():
                def make_listener(m=model):
                    last_state = ["idle"]
                    def _on_status_change(status):
                        task_id = f"model_dl_{m.id}"
                        prev = last_state[0]
                        if status.state == "downloading":
                            speed_str = f" ({status.speed_mb_s:.1f} MB/s)" if status.speed_mb_s > 0 else ""
                            eta_str = format_eta(status.eta_seconds)
                            status_text = f"{int(status.progress * 100)}%{speed_str}{eta_str}"
                            existing = self.get_task(task_id)
                            if not existing or not existing.is_running:
                                self.start_task(
                                    task_id=task_id,
                                    title=f"Downloading Model: {m.display_name}",
                                    initial_status=status_text,
                                    progress=status.progress,
                                    on_cancel=lambda: ModelDownloader.cancel_download(m.id),
                                    cancellable=True,
                                )
                            else:
                                self.update_task(task_id, status_text=status_text, progress=status.progress)
                        elif status.state == "completed" and prev == "downloading":
                            self.finish_task(
                                task_id=task_id,
                                completion_message=f"'{m.display_name}' downloaded successfully.",
                                notification_title="Model Download Completed",
                                show_toast=True,
                            )
                        elif status.state == "error" and prev == "downloading":
                            self.fail_task(
                                task_id=task_id,
                                error_message=status.error_message or "Error occurred during model download.",
                                notification_title="Model Download Error",
                                show_toast=True,
                            )
                        last_state[0] = status.state
                    return _on_status_change

                ModelDownloader.register_listener(model.id, make_listener(model))
        except Exception as ex:
            log_error(f"Error binding ModelDownloader to TaskNotificationManager: {ex}")
