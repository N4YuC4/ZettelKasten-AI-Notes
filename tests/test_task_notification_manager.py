# tests/test_task_notification_manager.py
#
# Comprehensive test suite for TaskNotificationManager, ToastOverlay,
# NotificationCenterBox, NotificationCenterButton, and AppController non-blocking background task workflows.

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import flet as ft

from task_notification_manager import TaskNotificationManager, NotificationItem, TaskItem
from ui.notification_center import ToastItem, ToastOverlay, NotificationCenterBox, NotificationCenterButton
from app_controller import AppController
from app_state import AppState


class TestTaskNotificationManager:
    """Tests for core TaskNotificationManager logic."""

    def test_manager_initial_state(self):
        mgr = TaskNotificationManager()
        assert mgr.get_active_tasks() == []
        assert mgr.get_active_task_count() == 0
        assert mgr.get_all_tasks() == []
        assert mgr.get_notifications() == []

    def test_start_task(self):
        mgr = TaskNotificationManager()
        task = mgr.start_task(
            task_id="task_1",
            title="Processing PDF",
            initial_status="Extracting pages...",
            cancellable=True
        )
        assert task.id == "task_1"
        assert task.title == "Processing PDF"
        assert task.status_text == "Extracting pages..."
        assert task.is_running is True
        assert task.cancellable is True
        assert mgr.get_active_task_count() == 1
        assert len(mgr.get_active_tasks()) == 1

    def test_update_task_progress_and_status(self):
        mgr = TaskNotificationManager()
        mgr.start_task("task_1", "Model Download")

        mgr.update_task("task_1", status_text="Downloading chunk 1...", progress=0.45)
        task = mgr.get_task("task_1")
        assert task.status_text == "Downloading chunk 1..."
        assert task.progress == 0.45

        # Progress clamped between 0.0 and 1.0
        mgr.update_task("task_1", progress=1.5)
        assert task.progress == 1.0

        mgr.update_task("task_1", progress=-0.5)
        assert task.progress == 0.0

    def test_finish_task(self):
        mgr = TaskNotificationManager()
        mgr.start_task("task_1", "Note Generation")

        toasts = []
        mgr.subscribe_toast(lambda n: toasts.append(n))

        mgr.finish_task("task_1", completion_message="5 notes generated.", show_toast=True)
        task = mgr.get_task("task_1")
        assert task.is_running is False
        assert task.progress == 1.0
        assert task.finished_at is not None
        assert mgr.get_active_task_count() == 0
        assert len(toasts) == 1
        assert toasts[0].notification_type == "success"
        assert "5 notes generated" in toasts[0].message

    def test_fail_task(self):
        mgr = TaskNotificationManager()
        mgr.start_task("task_1", "PDF Extraction")

        toasts = []
        mgr.subscribe_toast(lambda n: toasts.append(n))

        mgr.fail_task("task_1", error_message="File is corrupted.", show_toast=True)
        task = mgr.get_task("task_1")
        assert task.is_running is False
        assert task.error == "File is corrupted."
        assert mgr.get_active_task_count() == 0
        assert len(toasts) == 1
        assert toasts[0].notification_type == "error"
        assert "File is corrupted." in toasts[0].message

    def test_cancel_task_executes_callback(self):
        mgr = TaskNotificationManager()
        cancelled = []
        mgr.start_task(
            "task_1",
            "Long AI Task",
            on_cancel=lambda: cancelled.append(True)
        )

        mgr.cancel_task("task_1")
        assert cancelled == [True]
        task = mgr.get_task("task_1")
        assert task.is_running is False
        assert mgr.get_active_task_count() == 0

        # Warning notification generated
        notifs = mgr.get_notifications()
        assert len(notifs) >= 1
        assert notifs[0].notification_type == "warning"

    def test_remove_task(self):
        mgr = TaskNotificationManager()
        mgr.start_task("task_1", "Title")
        assert mgr.get_task("task_1") is not None

        mgr.remove_task("task_1")
        assert mgr.get_task("task_1") is None

    def test_add_notification_and_max_history(self):
        mgr = TaskNotificationManager(max_history=3)
        toasts = []
        mgr.subscribe_toast(lambda n: toasts.append(n))

        n1 = mgr.add_notification("Title 1", "Msg 1", notification_type="info")
        n2 = mgr.add_notification("Title 2", "Msg 2", notification_type="success")
        n3 = mgr.add_notification("Title 3", "Msg 3", notification_type="warning")
        n4 = mgr.add_notification("Title 4", "Msg 4", notification_type="error")

        assert len(toasts) == 4
        notifs = mgr.get_notifications()
        assert len(notifs) == 3
        assert notifs[0].title == "Title 4"
        assert notifs[1].title == "Title 3"
        assert notifs[2].title == "Title 2"

    def test_clear_and_remove_notification(self):
        mgr = TaskNotificationManager()
        n1 = mgr.add_notification("T1", "M1")
        n2 = mgr.add_notification("T2", "M2")
        assert len(mgr.get_notifications()) == 2

        mgr.remove_notification(n1.id)
        assert len(mgr.get_notifications()) == 1
        assert mgr.get_notifications()[0].id == n2.id

        mgr.clear_notifications()
        assert len(mgr.get_notifications()) == 0

    def test_subscribe_and_unsubscribe(self):
        mgr = TaskNotificationManager()
        calls = []
        listener = lambda: calls.append(True)

        mgr.subscribe(listener)
        mgr.add_notification("T", "M")
        assert len(calls) == 1

        mgr.unsubscribe(listener)
        mgr.add_notification("T2", "M2")
        assert len(calls) == 1


class TestNotificationCenterUI:
    """Tests for UI components in notification_center.py."""

    def test_toast_item_styling_for_types(self):
        for ntype, expected_icon in [
            ("success", ft.Icons.CHECK_CIRCLE),
            ("error", ft.Icons.ERROR_OUTLINE),
            ("warning", ft.Icons.WARNING_AMBER_ROUNDED),
            ("info", ft.Icons.INFO_OUTLINE),
        ]:
            notif = NotificationItem(title="Test", message="Body", notification_type=ntype)
            dismissed = []
            toast = ToastItem(notif, on_dismiss=lambda t: dismissed.append(t))

            assert toast.offset.x == 1.2  # Offscreen initial state
            assert toast.opacity == 0.0
            row = toast.content
            assert isinstance(row, ft.Row)
            icon_ctrl = row.controls[0]
            assert icon_ctrl.icon == expected_icon

    def test_toast_item_click_opens_box(self):
        notif = NotificationItem(title="Test", message="Body")
        box_opened = []
        dismissed = []
        toast = ToastItem(
            notif,
            on_dismiss=lambda t: dismissed.append(t),
            on_open_box=lambda: box_opened.append(True)
        )
        toast._handle_card_click()
        assert box_opened == [True]
        assert toast.dismiss_event.is_set()

    def test_toast_overlay_show_toast(self):
        page = MagicMock()
        page.run_task = MagicMock()
        overlay = ToastOverlay(page)

        notif = NotificationItem(title="Hello", message="World")
        overlay.show_toast(notif)

        assert len(overlay.column.controls) == 1
        page.run_task.assert_called_once()

    def test_notification_center_box_toggle_open_close(self):
        page = MagicMock()
        mgr = TaskNotificationManager()
        box = NotificationCenterBox(page, mgr)

        assert box.visible is False
        assert box.is_open is False

        box.toggle()
        assert box.is_open is True
        assert box.visible is True
        assert box.offset.x == 0
        assert box.offset.y == 0
        assert box.opacity == 1.0

        box.close_box()
        assert box.is_open is False
        assert box.opacity == 0.0

    def test_notification_center_box_renders_active_tasks_and_history(self):
        page = MagicMock()
        mgr = TaskNotificationManager()
        box = NotificationCenterBox(page, mgr)

        # Empty state check
        box.refresh()
        assert len(box.tasks_list_col.controls) == 1
        assert "no active tasks" in box.tasks_list_col.controls[0].content.controls[1].value.lower()
        assert len(box.notifications_list_col.controls) == 1
        assert "no notification history" in box.notifications_list_col.controls[0].content.controls[1].value.lower()

        # Add active task and notification
        task = mgr.start_task("t1", "PDF Note Generation", initial_status="Processing...", progress=0.5)
        mgr.add_notification("Note Saved", "Note added successfully.", notification_type="success")

        box.refresh()
        assert len(box.tasks_list_col.controls) == 1
        assert len(box.notifications_list_col.controls) == 1
        assert box.active_badge_chip.visible is True

    def test_notification_center_button_badges(self):
        mgr = TaskNotificationManager()
        btn = NotificationCenterButton(mgr)

        # Initially no badge
        assert btn.badge is None
        assert btn.icon == ft.Icons.NOTIFICATIONS_OUTLINED

        # Notification added -> badge updated
        mgr.add_notification("T", "M")
        assert btn.badge is not None
        assert btn.badge.label == "1"

        # Active task added -> badge turns to active count
        mgr.start_task("t1", "Task 1")
        assert btn.icon == ft.Icons.NOTIFICATIONS_ACTIVE
        assert btn.badge.label == "1"
        assert btn.badge.bgcolor == ft.Colors.AMBER_400


class TestAppControllerTaskIntegration:
    """Tests for AppController background worker & notification integration."""

    @pytest.fixture
    def controller_setup(self):
        page = MagicMock()
        page.overlay = []
        page.run_task = MagicMock()
        page.run_thread = MagicMock()
        page.update = MagicMock()

        db_manager = MagicMock()
        db_manager.db_path = ":memory:"
        db_manager.note_count.return_value = 0
        note_service = MagicMock()
        note_service.load_all_notes_metadata.return_value = ([], [])
        note_service.get_linked_note_ids.return_value = []
        dialog_manager = MagicMock()
        state = AppState()
        mgr = TaskNotificationManager()

        controller = AppController(
            page=page,
            db_manager=db_manager,
            note_service=note_service,
            dialog_manager=dialog_manager,
            state=state,
            task_notification_manager=mgr,
        )

        notification_box = NotificationCenterBox(page, mgr)
        toast_overlay = ToastOverlay(page, on_open_box=notification_box.open_box)
        notification_btn = NotificationCenterButton(mgr, on_click=notification_box.toggle)

        controller.attach_views(
            sidebar=MagicMock(),
            right_panel=MagicMock(),
            editor_workspace=MagicMock(),
            note_title_text=MagicMock(),
            notification_box=notification_box,
            toast_overlay=toast_overlay,
            notification_btn=notification_btn,
        )

        return controller, mgr, dialog_manager, page

    def test_controller_show_snack_bar_dispatches_toast_and_history(self, controller_setup):
        controller, mgr, dialog_manager, page = controller_setup

        controller.show_snack_bar("Note 'Quantum Physics' saved successfully.")

        notifs = mgr.get_notifications()
        assert len(notifs) == 1
        assert notifs[0].notification_type == "success"
        assert notifs[0].message == "Note 'Quantum Physics' saved successfully."

    def test_controller_cancel_worker_updates_task_manager(self, controller_setup):
        controller, mgr, dialog_manager, page = controller_setup

        mock_worker = MagicMock()
        controller.active_worker = mock_worker
        mgr.start_task("pdf_ai_generation", "PDF Note Generation")

        with patch("time.sleep"):
            controller.cancel_worker()

        assert controller.active_worker is None
        assert mgr.get_active_task_count() == 0
        dialog_manager.hide_loading.assert_called_once()

    def test_controller_handle_ai_finished_updates_task_manager(self, controller_setup):
        controller, mgr, dialog_manager, page = controller_setup
        mgr.start_task("pdf_ai_generation", "PDF Note Generation")

        with patch("time.sleep"):
            controller.handle_ai_finished([{"title": "Note 1"}])

        assert mgr.get_active_task_count() == 0
        dialog_manager.hide_loading.assert_called_once()
        notifs = mgr.get_notifications()
        assert any("1 notes successfully generated" in n.message for n in notifs)

    def test_controller_handle_ai_error_updates_task_manager(self, controller_setup):
        controller, mgr, dialog_manager, page = controller_setup
        mgr.start_task("pdf_ai_generation", "PDF Note Generation")

        with patch("time.sleep"):
            controller.handle_ai_error("Quota exceeded.")

        assert mgr.get_active_task_count() == 0
        dialog_manager.hide_loading.assert_called_once()
        task = mgr.get_task("pdf_ai_generation")
        assert task.error == "Quota exceeded."

    def test_trigger_pdf_generation_non_blocking_starts_background_task(self, controller_setup):
        controller, mgr, dialog_manager, page = controller_setup

        mock_file = MagicMock()
        mock_file.path = "/fake/sample_notes.pdf"
        controller.pdf_file_picker = MagicMock()

        async def fake_pick_files(**kwargs):
            return [mock_file]

        controller.pdf_file_picker.pick_files = fake_pick_files

        asyncio.run(controller.trigger_pdf_generation())

        # Verify that modal show_loading was NOT called (non-blocking!)
        dialog_manager.show_loading.assert_not_called()

        # Verify that task was registered in TaskNotificationManager
        assert mgr.get_active_task_count() == 1
        active_task = mgr.get_active_tasks()[0]
        assert "sample_notes.pdf" in active_task.title
        assert active_task.is_running is True

        # Verify background thread was launched
        page.run_thread.assert_called_once()

    def test_cancel_task_single_notification_cascade(self, controller_setup):
        """Verifies that cancelling an AI task with on_cancel=cancel_worker results in exactly 1 notification."""
        controller, mgr, dialog_manager, page = controller_setup

        mock_worker = MagicMock()
        controller.active_worker = mock_worker

        dispatched_toasts = []
        mgr.subscribe_toast(lambda n: dispatched_toasts.append(n))

        # Start task with on_cancel wired to controller.cancel_worker (simulating PDF worker setup)
        mgr.start_task(
            "pdf_ai_generation",
            "AI Note Generation: sample.pdf",
            on_cancel=controller.cancel_worker,
        )

        with patch("time.sleep"):
            # Trigger cancellation like clicking "Cancel" in UI box
            mgr.cancel_task("pdf_ai_generation")

        # Must have cancelled the worker
        mock_worker.cancel.assert_called_once()
        assert controller.active_worker is None
        assert mgr.get_active_task_count() == 0

        # Exactly 1 notification recorded and 1 toast dispatched (no triple notification bug!)
        notifs = mgr.get_notifications()
        assert len(notifs) == 1, f"Expected 1 notification but got {len(notifs)}: {[n.message for n in notifs]}"
        assert len(dispatched_toasts) == 1, f"Expected 1 toast but got {len(dispatched_toasts)}: {[t.message for t in dispatched_toasts]}"
        assert "cancelled" in notifs[0].message.lower() or "stopped" in notifs[0].message.lower()

    def test_cancel_worker_direct_call_single_notification(self, controller_setup):
        """Verifies direct cancel_worker call produces exactly 1 notification."""
        controller, mgr, dialog_manager, page = controller_setup

        mock_worker = MagicMock()
        controller.active_worker = mock_worker

        dispatched_toasts = []
        mgr.subscribe_toast(lambda n: dispatched_toasts.append(n))

        mgr.start_task(
            "pdf_ai_generation",
            "AI Note Generation: doc.pdf",
            on_cancel=controller.cancel_worker,
        )

        with patch("time.sleep"):
            controller.cancel_worker()

        notifs = mgr.get_notifications()
        assert len(notifs) == 1
        assert len(dispatched_toasts) == 1

