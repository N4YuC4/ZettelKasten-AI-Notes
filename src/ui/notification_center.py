# ui/notification_center.py
#
# Dedicated Notification & Running Tasks Center for Zettelkasten AI Notes.
# Implements:
# 1. Corner floating toast overlay with smooth slide-in / slide-out animations.
# 2. Activity Center drawer/panel displaying ongoing background tasks (with live status,
#    progress indicators, and cancel controls) and persistent notification history.
# 3. AppBar action toggle button with dynamic badge counters and running task indicators.

import asyncio
from typing import Optional, Callable, Dict, Any, List
import flet as ft

from task_notification_manager import TaskNotificationManager, NotificationItem, TaskItem
from logger import log_debug, log_error


class ToastItem(ft.Container):
    """
    Individual floating toast card with slide animation support.
    """

    def __init__(
        self,
        notification: NotificationItem,
        on_dismiss: Callable[["ToastItem"], None],
        on_open_box: Optional[Callable[[], None]] = None,
    ):
        self.notification = notification
        self.on_dismiss_cb = on_dismiss
        self.on_open_box_cb = on_open_box
        self.dismiss_event = asyncio.Event()

        # Visual color and icon mapping
        ntype = notification.notification_type
        if ntype == "success":
            icon_name = ft.Icons.CHECK_CIRCLE
            accent_color = ft.Colors.GREEN_400
        elif ntype == "error":
            icon_name = ft.Icons.ERROR_OUTLINE
            accent_color = ft.Colors.RED_400
        elif ntype == "warning":
            icon_name = ft.Icons.WARNING_AMBER_ROUNDED
            accent_color = ft.Colors.AMBER_400
        else:
            icon_name = ft.Icons.INFO_OUTLINE
            accent_color = ft.Colors.PRIMARY

        icon_ctrl = ft.Icon(icon_name, color=accent_color, size=20)
        title_ctrl = ft.Text(
            notification.title or "Notification",
            size=12,
            weight=ft.FontWeight.BOLD,
            color=accent_color,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
        time_ctrl = ft.Text(
            notification.timestamp,
            size=10,
            color=ft.Colors.ON_SURFACE_VARIANT,
        )
        msg_ctrl = ft.Text(
            notification.message,
            size=11,
            color=ft.Colors.ON_SURFACE,
            max_lines=3,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        close_btn = ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_size=14,
            icon_color=ft.Colors.ON_SURFACE_VARIANT,
            tooltip="Close",
            on_click=lambda e: self.trigger_dismiss(),
            style=ft.ButtonStyle(padding=ft.Padding.all(2)),
        )

        card_content = ft.Row([
            icon_ctrl,
            ft.Column([
                ft.Row([title_ctrl, time_ctrl], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                msg_ctrl,
            ], spacing=2, expand=True),
            close_btn,
        ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        super().__init__(
            content=card_content,
            width=360,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=10,
            border=ft.Border.all(1.2, accent_color),
            shadow=ft.BoxShadow(
                blur_radius=12,
                spread_radius=1,
                color=ft.Colors.with_opacity(0.35, ft.Colors.BLACK),
                offset=ft.Offset(0, 4),
            ),
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            offset=ft.Offset(1.2, 0),  # Starts completely off-screen to the right
            animate_offset=ft.Animation(350, ft.AnimationCurve.DECELERATE),
            opacity=0.0,
            animate_opacity=ft.Animation(250),
            on_click=lambda e: self._handle_card_click(),
            tooltip="Click to view details",
        )

    def _handle_card_click(self):
        """Clicking on toast opens the task & notification center."""
        if self.on_open_box_cb:
            try:
                self.on_open_box_cb()
            except Exception:
                pass
        self.trigger_dismiss()

    def trigger_dismiss(self):
        """Signals early dismissal of this toast."""
        self.dismiss_event.set()


class ToastOverlay(ft.Container):
    """
    Floating container anchored at the bottom-right corner of the window
    managing sliding animated toast notifications.
    """

    def __init__(self, page: ft.Page, on_open_box: Optional[Callable[[], None]] = None):
        self.host_page = page
        self.on_open_box = on_open_box
        self.column = ft.Column(spacing=8, tight=True)

        super().__init__(
            bottom=20,
            right=20,
            width=370,
            content=self.column,
        )

    def show_toast(self, notification: NotificationItem) -> None:
        """Instantiates and plays slide-in / slide-out animation for a toast."""
        toast = ToastItem(
            notification=notification,
            on_dismiss=self._remove_toast,
            on_open_box=self.on_open_box,
        )
        self.column.controls.append(toast)
        try:
            self.column.update()
        except Exception:
            try:
                self.host_page.update()
            except Exception:
                pass

        if hasattr(self.host_page, "run_task") and callable(self.host_page.run_task):
            try:
                self.host_page.run_task(self._animate_toast_lifecycle, toast, notification.duration)
                return
            except Exception:
                pass

        # Fallback if run_task fails or in tests
        toast.offset = ft.Offset(0, 0)
        toast.opacity = 1.0

    async def _animate_toast_lifecycle(self, toast: ToastItem, duration: float) -> None:
        """Asynchronously slides in, waits, then slides out the toast."""
        try:
            # 1. Slide In Animation
            await asyncio.sleep(0.03)
            toast.offset = ft.Offset(0, 0)
            toast.opacity = 1.0
            try:
                toast.update()
            except Exception:
                pass

            # 2. Stay on screen for specified duration or until dismissed
            try:
                await asyncio.wait_for(toast.dismiss_event.wait(), timeout=duration)
            except asyncio.TimeoutError:
                pass

            # 3. Slide Out Animation
            toast.offset = ft.Offset(1.2, 0)
            toast.opacity = 0.0
            try:
                toast.update()
            except Exception:
                pass

            await asyncio.sleep(0.35)
            self._remove_toast(toast)
        except Exception as ex:
            log_error(f"Error in toast animation lifecycle: {ex}")
            self._remove_toast(toast)

    def _remove_toast(self, toast: ToastItem) -> None:
        """Removes toast from UI column."""
        if toast in self.column.controls:
            self.column.controls.remove(toast)
            try:
                self.column.update()
            except Exception:
                try:
                    self.host_page.update()
                except Exception:
                    pass


class NotificationCenterBox(ft.Container):
    """
    Floating panel / drawer  in top-right displaying:
    - Ongoing background tasks (live status, progress bar, cancel action)
    - Notification history (type, timestamp, message, clear actions)
    """

    def __init__(
        self,
        page: ft.Page,
        manager: TaskNotificationManager,
        on_close: Optional[Callable[[], None]] = None,
    ):
        self.host_page = page
        self.manager = manager
        self.on_close_cb = on_close
        self.is_open = False

        # Header controls
        self.title_text = ft.Text(
            "Tasks & Notifications",
            size=14,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.ON_SURFACE,
        )
        self.active_badge_chip = ft.Container(
            content=ft.Row([
                ft.ProgressRing(width=10, height=10, stroke_width=2, color=ft.Colors.PRIMARY),
                ft.Text("0 Active", size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
            ], spacing=4),
            padding=ft.Padding.symmetric(horizontal=6, vertical=2),
            bgcolor=ft.Colors.PRIMARY_CONTAINER,
            border_radius=10,
            visible=False,
        )
        self.clear_btn = ft.IconButton(
            icon=ft.Icons.DELETE_SWEEP_OUTLINED,
            icon_size=18,
            icon_color=ft.Colors.ON_SURFACE_VARIANT,
            tooltip="Clear Notification History",
            on_click=lambda e: self._handle_clear_history(),
        )
        self.close_btn = ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_size=18,
            icon_color=ft.Colors.ON_SURFACE_VARIANT,
            tooltip="Close",
            on_click=lambda e: self.close_box(),
        )

        header_row = ft.Row([
            ft.Row([
                ft.Icon(ft.Icons.TASK_ALT_ROUNDED, color=ft.Colors.PRIMARY, size=18),
                self.title_text,
                self.active_badge_chip,
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([
                self.clear_btn,
                self.close_btn,
            ], spacing=2),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        # Section lists
        self.tasks_list_col = ft.Column(spacing=6)
        self.notifications_list_col = ft.Column(spacing=6)

        # Main scrollable body
        content_column = ft.Column([
            header_row,
            ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
            ft.Container(
                content=ft.Column([
                    # 1. Running Tasks Section
                    ft.Row([
                        ft.Text("Running Tasks", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                    ], alignment=ft.MainAxisAlignment.START),
                    self.tasks_list_col,
                    ft.Divider(height=16, color=ft.Colors.OUTLINE_VARIANT),

                    # 2. Notification History Section
                    ft.Row([
                        ft.Text("Notification History", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                    ], alignment=ft.MainAxisAlignment.START),
                    self.notifications_list_col,
                ], spacing=8, scroll=ft.ScrollMode.AUTO),
                expand=True,
            ),
        ], spacing=8, expand=True)

        super().__init__(
            top=60,
            right=16,
            width=390,
            height=490,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
            border_radius=14,
            border=ft.Border.all(1.2, ft.Colors.OUTLINE_VARIANT),
            shadow=ft.BoxShadow(
                blur_radius=20,
                spread_radius=2,
                color=ft.Colors.with_opacity(0.4, ft.Colors.BLACK),
                offset=ft.Offset(0, 6),
            ),
            padding=12,
            visible=False,
            offset=ft.Offset(0, -0.03),
            animate_offset=ft.Animation(250, ft.AnimationCurve.DECELERATE),
            opacity=0.0,
            animate_opacity=ft.Animation(200),
            content=content_column,
        )

        # Subscribe to manager events to keep view reactive
        self.manager.subscribe(self.refresh)

    def _handle_clear_history(self):
        """Clears notification history."""
        self.manager.clear_notifications()
        self.refresh()

    def toggle(self) -> None:
        """Toggles visibility of the notification & task box."""
        if self.is_open:
            self.close_box()
        else:
            self.open_box()

    def open_box(self) -> None:
        """Smoothly opens the panel and renders latest state."""
        self.is_open = True
        self.visible = True
        self.refresh()
        self.offset = ft.Offset(0, 0)
        self.opacity = 1.0
        try:
            self.update()
        except Exception:
            try:
                self.host_page.update()
            except Exception:
                pass

    def close_box(self) -> None:
        """Smoothly closes the panel."""
        self.is_open = False
        self.offset = ft.Offset(0, -0.03)
        self.opacity = 0.0
        try:
            self.update()
        except Exception:
            pass

        async def _hide_after_anim():
            await asyncio.sleep(0.25)
            if not self.is_open:
                self.visible = False
                try:
                    self.update()
                except Exception:
                    pass

        if hasattr(self.host_page, "run_task") and callable(self.host_page.run_task):
            try:
                self.host_page.run_task(_hide_after_anim)
            except Exception:
                self.visible = False
        else:
            self.visible = False

        if self.on_close_cb:
            try:
                self.on_close_cb()
            except Exception:
                pass

    def refresh(self) -> None:
        """Re-renders running tasks and notification history."""
        active_tasks = self.manager.get_active_tasks()
        active_count = len(active_tasks)

        # 1. Update Header Chip
        if active_count > 0:
            self.active_badge_chip.visible = True
            row_ctrl = self.active_badge_chip.content
            if isinstance(row_ctrl, ft.Row) and len(row_ctrl.controls) > 1:
                row_ctrl.controls[1].value = f"{active_count} Active"
        else:
            self.active_badge_chip.visible = False

        # 2. Render Active Tasks
        self.tasks_list_col.controls.clear()
        if not active_tasks:
            self.tasks_list_col.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, size=14, color=ft.Colors.ON_SURFACE_VARIANT),
                        ft.Text("No active tasks currently running.", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ], spacing=6),
                    padding=ft.Padding.symmetric(vertical=4, horizontal=6),
                )
            )
        else:
            for task in active_tasks:
                card = self._build_task_card(task)
                self.tasks_list_col.controls.append(card)

        # 3. Render Notification History
        notifications = self.manager.get_notifications()
        self.notifications_list_col.controls.clear()
        if not notifications:
            self.notifications_list_col.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.NOTIFICATIONS_NONE, size=14, color=ft.Colors.ON_SURFACE_VARIANT),
                        ft.Text("No notification history yet.", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ], spacing=6),
                    padding=ft.Padding.symmetric(vertical=4, horizontal=6),
                )
            )
        else:
            for notif in notifications:
                card = self._build_notification_card(notif)
                self.notifications_list_col.controls.append(card)

        if self.is_open:
            try:
                self.update()
            except Exception:
                try:
                    self.host_page.update()
                except Exception:
                    pass

    def _build_task_card(self, task: TaskItem) -> ft.Container:
        """Constructs UI card for an ongoing task."""
        progress_val = task.progress

        progress_bar = ft.ProgressBar(
            value=progress_val,
            color=ft.Colors.PRIMARY,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            height=4,
        )

        cancel_btn = (
            ft.TextButton(
                "Cancel",
                icon=ft.Icons.CANCEL,
                icon_color=ft.Colors.ERROR,
                style=ft.ButtonStyle(
                    color=ft.Colors.ERROR,
                    padding=ft.Padding.symmetric(horizontal=8, vertical=2),
                ),
                tooltip="Cancel Task",
                on_click=lambda e, tid=task.id: self.manager.cancel_task(tid),
            )
            if task.cancellable
            else ft.Container()
        )

        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.ProgressRing(width=14, height=14, stroke_width=2.5, color=ft.Colors.PRIMARY),
                    ft.Text(task.title, size=12, weight=ft.FontWeight.BOLD, expand=True, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                    cancel_btn,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Text(task.status_text or "Processing...", size=11, color=ft.Colors.ON_SURFACE_VARIANT, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                progress_bar,
            ], spacing=4),
            padding=8,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border_radius=8,
            border=ft.Border.all(1, ft.Colors.PRIMARY_CONTAINER),
        )

    def _build_notification_card(self, notif: NotificationItem) -> ft.Container:
        """Constructs UI card for a historical notification item."""
        ntype = notif.notification_type
        if ntype == "success":
            icon = ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN_400, size=16)
            border_color = ft.Colors.GREEN_900
        elif ntype == "error":
            icon = ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.RED_400, size=16)
            border_color = ft.Colors.RED_900
        elif ntype == "warning":
            icon = ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.AMBER_400, size=16)
            border_color = ft.Colors.AMBER_900
        else:
            icon = ft.Icon(ft.Icons.INFO_OUTLINE, color=ft.Colors.PRIMARY, size=16)
            border_color = ft.Colors.PRIMARY_CONTAINER

        return ft.Container(
            content=ft.Row([
                icon,
                ft.Column([
                    ft.Row([
                        ft.Text(notif.title or "Notification", size=11, weight=ft.FontWeight.W_600, expand=True),
                        ft.Text(notif.timestamp, size=9, color=ft.Colors.ON_SURFACE_VARIANT),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Text(notif.message, size=11, color=ft.Colors.ON_SURFACE_VARIANT, max_lines=3, overflow=ft.TextOverflow.ELLIPSIS),
                ], spacing=2, expand=True),
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_size=12,
                    icon_color=ft.Colors.ON_SURFACE_VARIANT,
                    tooltip="Remove",
                    on_click=lambda e, nid=notif.id: self.manager.remove_notification(nid),
                    style=ft.ButtonStyle(padding=ft.Padding.all(2)),
                ),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.symmetric(horizontal=8, vertical=6),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border_radius=8,
            border=ft.Border.all(0.8, border_color),
        )


class NotificationCenterButton(ft.IconButton):
    """
    AppBar action button with reactive badge indicators for running tasks & unread notifications.
    """

    def __init__(
        self,
        manager: TaskNotificationManager,
        on_click: Optional[Callable[[], None]] = None,
    ):
        self.manager = manager
        self.on_click_callback = on_click

        super().__init__(
            icon=ft.Icons.NOTIFICATIONS_OUTLINED,
            tooltip="Running Tasks & Notifications",
            on_click=lambda e: self._handle_click(),
        )

        self.manager.subscribe(self.update_badge_state)
        self.update_badge_state()

    def _handle_click(self):
        if self.on_click_callback:
            self.on_click_callback()

    def update_badge_state(self):
        """Updates the icon badge and color depending on running tasks."""
        active_count = self.manager.get_active_task_count()
        notif_count = len(self.manager.get_notifications())

        if active_count > 0:
            self.icon = ft.Icons.NOTIFICATIONS_ACTIVE
            self.icon_color = ft.Colors.AMBER_400
            self.badge = ft.Badge(
                label=str(active_count),
                bgcolor=ft.Colors.AMBER_400,
                text_color=ft.Colors.BLACK,
            )
            self.tooltip = f"Running Tasks & Notifications ({active_count} active task{'s' if active_count > 1 else ''} running in background)"
        elif notif_count > 0:
            self.icon = ft.Icons.NOTIFICATIONS_OUTLINED
            self.icon_color = ft.Colors.PRIMARY
            self.badge = ft.Badge(
                label=str(min(notif_count, 99)),
                bgcolor=ft.Colors.PRIMARY,
                text_color=ft.Colors.ON_PRIMARY,
            )
            self.tooltip = f"Running Tasks & Notifications ({notif_count} notification{'s' if notif_count > 1 else ''})"
        else:
            self.icon = ft.Icons.NOTIFICATIONS_OUTLINED
            self.icon_color = None
            self.badge = None
            self.tooltip = "Running Tasks & Notifications"

        try:
            self.update()
        except Exception:
            pass
