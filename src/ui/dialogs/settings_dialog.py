"""
Settings dialog builder for Zettelkasten AI Notes.
Constructs a tabbed settings modal for configuring AI providers,
storage directories, hardware acceleration, and system diagnostics.
"""

import os
import inspect
from typing import Any, Callable, Optional, TYPE_CHECKING
import flet as ft

from logger import log_error
from hardware_checker import HardwareChecker
import local_models_catalog

if TYPE_CHECKING:
    from ui.dialog_manager import DialogManager


def build_settings_dialog(
    dm: "DialogManager",
    theme_btn: ft.IconButton,
    auto_save_switch: ft.Switch,
    current_ai_provider: str,
    current_active_model_id: str,
    on_save_settings: Optional[Callable[..., None]] = None,
    on_open_model_manager: Optional[Callable[[], None]] = None,
    current_gpu_acceleration: bool = True,
    current_api_key: str = "",
    current_db_path: str = "",
    current_models_dir: str = "",
    current_custom_prompt: str = "",
    current_confirm_delete_note: bool = True,
    current_confirm_delete_col: bool = True,
    current_custom_db_path: str = "",
    on_reset_defaults: Optional[Callable[[], None]] = None,
    on_setting_changed: Optional[Callable[[str, Any], None]] = None,
) -> None:
    """Builds and opens the comprehensive 4-tab settings dialog."""
    selected_provider = current_ai_provider or "gemini"
    selected_model_id = current_active_model_id or local_models_catalog.DEFAULT_MODEL_ID

    def persist_change(key: str, value: Any):
        str_val = str(value) if value is not None else ""
        if on_setting_changed:
            try:
                on_setting_changed(key, str_val)
            except Exception as ex:
                log_error(f"Error auto-saving setting '{key}': {ex}")
        if on_save_settings:
            try:
                sig = inspect.signature(on_save_settings)
                if len(sig.parameters) <= 4:
                    on_save_settings(
                        (dm.api_key_field.value or "").strip(),
                        provider_dropdown.value or "gemini",
                        selected_model_id,
                        bool(gpu_switch.value)
                    )
                else:
                    on_save_settings(
                        api_key=(dm.api_key_field.value or "").strip(),
                        ai_provider=provider_dropdown.value or "gemini",
                        active_model_id=selected_model_id,
                        gpu_acceleration=bool(gpu_switch.value),
                        models_dir=(models_dir_field.value or "").strip(),
                        custom_prompt=(custom_prompt_field.value or "").strip(),
                        confirm_delete_note=bool(confirm_note_switch.value),
                        confirm_delete_collection=bool(confirm_col_switch.value),
                        custom_db_path=(custom_db_field.value or "").strip(),
                    )
            except Exception as ex:
                log_error(f"Error delegating save for setting '{key}': {ex}")

    # Controls for Tab 1: General
    confirm_note_switch = ft.Switch(
        value=current_confirm_delete_note,
        tooltip="Prompt for confirmation before permanently deleting a note.",
        on_change=lambda e: persist_change("CONFIRM_DELETE_NOTE", "True" if e.control.value else "False")
    )
    confirm_col_switch = ft.Switch(
        value=current_confirm_delete_col,
        tooltip="Prompt for confirmation before deleting a collection and all its notes.",
        on_change=lambda e: persist_change("CONFIRM_DELETE_COLLECTION", "True" if e.control.value else "False")
    )

    # Controls for Tab 2: AI
    dm.api_key_field.value = current_api_key or ""
    dm.api_key_field.on_change = lambda e: persist_change("GEMINI_API_KEY", (dm.api_key_field.value or "").strip())
    dm.api_key_field.on_blur = lambda e: persist_change("GEMINI_API_KEY", (dm.api_key_field.value or "").strip())

    gpu_switch = ft.Switch(
        value=current_gpu_acceleration,
        tooltip="When enabled, uses graphics card (Vulkan / CUDA) for faster inference. When disabled, uses CPU only.",
        on_change=lambda e: persist_change("GPU_ACCELERATION", "True" if e.control.value else "False")
    )
    custom_prompt_field = ft.TextField(
        label="Custom Extraction Instructions (Optional)",
        value=current_custom_prompt or "",
        multiline=True,
        min_lines=3,
        max_lines=5,
        border_radius=8,
        hint_text="e.g. Always cite page numbers, use bullet points, or focus on empirical methodology...",
        on_change=lambda e: persist_change("AI_CUSTOM_SYSTEM_PROMPT", (custom_prompt_field.value or "").strip()),
        on_blur=lambda e: persist_change("AI_CUSTOM_SYSTEM_PROMPT", (custom_prompt_field.value or "").strip()),
    )

    provider_details_container = ft.Container()

    def update_provider_view(provider_value: str):
        nonlocal selected_provider
        selected_provider = provider_value
        if selected_provider == "local":
            model_info = local_models_catalog.get_model_by_id(selected_model_id)
            display_title = model_info.display_name if model_info else selected_model_id
            accel_info = HardwareChecker.get_acceleration_info()
            accel_badge = ft.Row([
                ft.Icon(
                    ft.Icons.BOLT if accel_info["gpu_offload_supported"] else ft.Icons.INFO_OUTLINE,
                    size=14,
                    color=ft.Colors.GREEN_400 if accel_info["gpu_offload_supported"] else ft.Colors.AMBER_400
                ),
                ft.Text(
                    f"Hardware Support: {accel_info['active_backend']}",
                    size=11,
                    color=ft.Colors.GREEN_400 if accel_info["gpu_offload_supported"] else ft.Colors.AMBER_400,
                    weight=ft.FontWeight.W_500
                )
            ], spacing=4)

            gpu_setting_row = ft.Row([
                ft.Column([
                    ft.Text("GPU Hardware Acceleration", weight=ft.FontWeight.W_600, size=13),
                    ft.Text(
                        "Accelerates note extraction using the graphics card (Vulkan / CUDA).",
                        size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT
                    )
                ], expand=True, spacing=1),
                gpu_switch
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER)

            provider_details_container.content = ft.Column([
                ft.Text("Local GGUF Model Configuration", weight=ft.FontWeight.BOLD),
                ft.Text("Model runs directly on your device; no data is sent externally.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Container(
                    content=ft.Row([
                        ft.Column([
                            ft.Text(f"Active Model: {display_title}", weight=ft.FontWeight.W_600),
                            ft.Text(model_info.user_description if model_info else "", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                            accel_badge
                        ], expand=True, spacing=2),
                        ft.Button(
                            "Model Manager",
                            icon=ft.Icons.MEMORY,
                            on_click=lambda e: (dm.close(dm.settings_dialog), on_open_model_manager())
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=10,
                    bgcolor=ft.Colors.SURFACE_CONTAINER,
                    border_radius=8
                ),
                gpu_setting_row
            ], spacing=10)
        else:
            provider_details_container.content = ft.Column([
                ft.Text("Gemini API Configuration", weight=ft.FontWeight.BOLD),
                ft.Text("API key is securely stored in your dedicated settings database.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                dm.api_key_field
            ], spacing=10)
        try:
            dm.settings_dialog.update()
        except Exception:
            pass

    provider_dropdown = ft.Dropdown(
        label="AI Provider",
        value=selected_provider,
        options=[
            ft.dropdown.Option(key="gemini", text="Google Gemini (Cloud)"),
            ft.dropdown.Option(key="local", text="Local GGUF Model (On-Device / Offline)"),
        ],
        border_radius=8,
        on_select=lambda e: (
            update_provider_view(provider_dropdown.value or "gemini"),
            persist_change("AI_PROVIDER", provider_dropdown.value or "gemini")
        )
    )
    update_provider_view(selected_provider)

    # Controls for Tab 3: Storage
    active_db = current_db_path or os.path.abspath(os.path.join("db", "notes.db"))
    db_size_kb = 0.0
    if os.path.exists(active_db):
        try:
            db_size_kb = os.path.getsize(active_db) / 1024.0
        except Exception:
            pass

    custom_db_field = ft.TextField(
        label="Custom Database Path (Optional)",
        value=current_custom_db_path or "",
        hint_text="Default: db/notes.db",
        border_radius=8,
        on_change=lambda e: persist_change("CUSTOM_DB_PATH", (custom_db_field.value or "").strip()),
        on_blur=lambda e: persist_change("CUSTOM_DB_PATH", (custom_db_field.value or "").strip()),
    )

    models_dir_field = ft.TextField(
        label="Local Models Storage Directory",
        value=current_models_dir or local_models_catalog.get_default_models_dir(),
        border_radius=8,
        on_change=lambda e: persist_change("MODELS_DIR", (models_dir_field.value or "").strip()),
        on_blur=lambda e: persist_change("MODELS_DIR", (models_dir_field.value or "").strip()),
    )

    # Controls for Tab 4: System
    mem_info = HardwareChecker.get_system_memory_info()
    cpu_info = HardwareChecker.get_cpu_info()
    accel_info = HardwareChecker.get_acceleration_info()

    hardware_card = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.ANALYTICS_OUTLINED, color=ft.Colors.PRIMARY, size=20),
                ft.Text("Live System Diagnostics", weight=ft.FontWeight.BOLD, size=13),
            ], spacing=8),
            ft.Divider(height=1),
            ft.Row([
                ft.Icon(ft.Icons.MEMORY, size=16, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(f"CPU: {cpu_info['physical_cores']} Cores / {cpu_info['logical_cores']} Threads (Optimal: {cpu_info['optimal_threads']})", size=12),
            ], spacing=6),
            ft.Row([
                ft.Icon(ft.Icons.STORAGE, size=16, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(f"RAM: Available {mem_info['available_gb']:.1f} GB / Total {mem_info['total_gb']:.1f} GB", size=12),
            ], spacing=6),
            ft.Row([
                ft.Icon(
                    ft.Icons.BOLT if accel_info["gpu_offload_supported"] else ft.Icons.INFO_OUTLINE,
                    size=16,
                    color=ft.Colors.GREEN_400 if accel_info["gpu_offload_supported"] else ft.Colors.AMBER_400
                ),
                ft.Text(f"Acceleration: {accel_info['active_backend']}", size=12, color=ft.Colors.GREEN_400 if accel_info["gpu_offload_supported"] else ft.Colors.AMBER_400, weight=ft.FontWeight.W_500),
            ], spacing=6),
        ], spacing=6),
        padding=12,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        border_radius=8
    )

    shortcuts_card = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.KEYBOARD, color=ft.Colors.PRIMARY, size=20),
                ft.Text("Keyboard Shortcuts Reference", weight=ft.FontWeight.BOLD, size=13),
            ], spacing=8),
            ft.Divider(height=1),
            ft.Row([ft.Text("Ctrl + N", weight=ft.FontWeight.BOLD, size=12), ft.Text("Create New Note", size=12, color=ft.Colors.ON_SURFACE_VARIANT)], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Text("Ctrl + S", weight=ft.FontWeight.BOLD, size=12), ft.Text("Save Current Note", size=12, color=ft.Colors.ON_SURFACE_VARIANT)], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Text("Ctrl + E", weight=ft.FontWeight.BOLD, size=12), ft.Text("Toggle Source / Reading Mode", size=12, color=ft.Colors.ON_SURFACE_VARIANT)], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Text("Ctrl + K", weight=ft.FontWeight.BOLD, size=12), ft.Text("Insert WikiLink [[...]]", size=12, color=ft.Colors.ON_SURFACE_VARIANT)], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Text("Ctrl + B / I", weight=ft.FontWeight.BOLD, size=12), ft.Text("Format Bold / Italic", size=12, color=ft.Colors.ON_SURFACE_VARIANT)], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Text("Ctrl + [ / ]", weight=ft.FontWeight.BOLD, size=12), ft.Text("Toggle Left / Right Panels", size=12, color=ft.Colors.ON_SURFACE_VARIANT)], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ], spacing=5),
        padding=12,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        border_radius=8
    )

    tab_container = ft.Container(height=340)

    def switch_tab(tab_name: str):
        if tab_name == "general":
            tab_container.content = ft.Column([
                ft.Row([
                    ft.Column([
                        ft.Text("Theme Mode", weight=ft.FontWeight.W_600, size=13),
                        ft.Text("Toggle between Dark and Light mode.", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ], expand=True, spacing=1),
                    theme_btn
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=1),
                ft.Row([
                    ft.Column([
                        ft.Text("Auto Save", weight=ft.FontWeight.W_600, size=13),
                        ft.Text("Automatically save note edits periodically.", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ], expand=True, spacing=1),
                    auto_save_switch
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=1),
                ft.Row([
                    ft.Column([
                        ft.Text("Confirm Note Deletion", weight=ft.FontWeight.W_600, size=13),
                        ft.Text("Display confirmation prompt before deleting a note.", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ], expand=True, spacing=1),
                    confirm_note_switch
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=1),
                ft.Row([
                    ft.Column([
                        ft.Text("Confirm Collection Deletion", weight=ft.FontWeight.W_600, size=13),
                        ft.Text("Display confirmation prompt before deleting an entire collection.", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ], expand=True, spacing=1),
                    confirm_col_switch
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=12, scroll=ft.ScrollMode.AUTO)
        elif tab_name == "ai":
            tab_container.content = ft.Column([
                provider_dropdown,
                provider_details_container,
                ft.Divider(height=1),
                ft.Column([
                    ft.Text("Custom System Prompt (Optional)", weight=ft.FontWeight.W_600, size=13),
                    custom_prompt_field,
                    ft.Text(
                        "Rule Precedence: Core Zettelkasten rules (valid JSON, language matching, atomic scope) strictly supersede any conflicting user instructions.",
                        size=11,
                        color=ft.Colors.AMBER_400,
                        italic=True
                    ),
                ], spacing=6)
            ], spacing=12, scroll=ft.ScrollMode.AUTO)
        elif tab_name == "storage":
            tab_container.content = ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.STORAGE, color=ft.Colors.PRIMARY, size=20),
                            ft.Text("Active Notes Database", weight=ft.FontWeight.BOLD, size=13),
                            ft.Container(
                                content=ft.Text(f"{db_size_kb:.1f} KB", size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                                bgcolor=ft.Colors.PRIMARY_CONTAINER,
                                border_radius=8,
                                padding=ft.Padding.symmetric(horizontal=6, vertical=2)
                            )
                        ], spacing=8),
                        ft.Text(active_db, size=11, color=ft.Colors.ON_SURFACE_VARIANT, selectable=True),
                    ], spacing=4),
                    padding=10,
                    bgcolor=ft.Colors.SURFACE_CONTAINER,
                    border_radius=8
                ),
                custom_db_field,
                ft.Divider(height=1),
                models_dir_field,
                ft.TextButton(
                    "Reset Models Directory to Default",
                    icon=ft.Icons.RESTORE,
                    on_click=lambda e: (
                        setattr(models_dir_field, "value", local_models_catalog.get_default_models_dir()),
                        models_dir_field.update(),
                        persist_change("MODELS_DIR", local_models_catalog.get_default_models_dir())
                    )
                )
            ], spacing=12, scroll=ft.ScrollMode.AUTO)
        elif tab_name == "system":
            tab_container.content = ft.Column([
                hardware_card,
                shortcuts_card,
                ft.Divider(height=1),
                ft.Row([
                    ft.Column([
                        ft.Text("Factory Reset", weight=ft.FontWeight.BOLD, size=13, color=ft.Colors.ERROR),
                        ft.Text("Restore all settings in settings.db to defaults.", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ], expand=True, spacing=1),
                    ft.Button(
                        "Reset to Defaults",
                        icon=ft.Icons.RESTART_ALT,
                        style=ft.ButtonStyle(color=ft.Colors.ERROR),
                        on_click=lambda e: dm.show_reset_settings_confirm(
                            on_confirm=lambda: (
                                dm.close(dm.settings_dialog),
                                on_reset_defaults() if on_reset_defaults else None
                            )
                        )
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER)
            ], spacing=12, scroll=ft.ScrollMode.AUTO)

        try:
            dm.settings_dialog.update()
        except Exception:
            pass

    tab_segmented = ft.SegmentedButton(
        selected=["general"],
        allow_multiple_selection=False,
        show_selected_icon=False,
        segments=[
            ft.Segment(value="general", label=ft.Text("General", size=12), icon=ft.Icon(ft.Icons.TUNE, size=16)),
            ft.Segment(value="ai", label=ft.Text("AI", size=12), icon=ft.Icon(ft.Icons.AUTO_AWESOME, size=16)),
            ft.Segment(value="storage", label=ft.Text("Storage", size=12), icon=ft.Icon(ft.Icons.FOLDER_OPEN, size=16)),
            ft.Segment(value="system", label=ft.Text("System", size=12), icon=ft.Icon(ft.Icons.COMPUTER, size=16)),
        ],
        on_change=lambda e: switch_tab(list(e.control.selected)[0] if e.control.selected else "general")
    )

    switch_tab("general")

    dm.settings_dialog.content = ft.Column([
        tab_segmented,
        ft.Divider(height=1),
        tab_container,
        ft.Divider(height=1),
        ft.Row([
            ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, size=14, color=ft.Colors.GREEN_400),
            ft.Text("Settings are saved automatically", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        ], spacing=6)
    ], tight=True, spacing=10, width=580)

    dm.settings_dialog.actions = [
        ft.TextButton(
            "Close",
            icon=ft.Icons.CLOSE,
            on_click=lambda e: dm.close(dm.settings_dialog)
        )
    ]
    dm._safe_open(dm.settings_dialog)

