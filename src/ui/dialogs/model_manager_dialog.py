"""
Model Manager dialog builder for Zettelkasten AI Notes.
Constructs the model catalog modal, displaying curated GGUF models,
hardware diagnostics, memory unload controls, and async download tracking.
"""

import asyncio
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING
import flet as ft

from hardware_checker import HardwareChecker
import local_models_catalog
from model_downloader import ModelDownloader
from local_gguf_client import LocalGgufClient

if TYPE_CHECKING:
    from ui.dialog_manager import DialogManager


def build_model_manager_dialog(
    dm: "DialogManager",
    models_dir: str,
    active_model_id: str,
    on_select_model: Callable[[str], None],
    on_model_deleted: Optional[Callable[[str], None]] = None,
) -> None:
    """Builds and opens the Model Manager dialog."""
    mem_info = HardwareChecker.get_system_memory_info()
    cpu_info = HardwareChecker.get_cpu_info()
    accel_info = HardwareChecker.get_acceleration_info()

    models_listview = ft.ListView(spacing=10, height=420, expand=True)
    model_controls_map: Dict[str, Tuple[ft.ProgressBar, ft.Text]] = {}
    is_dialog_active = True

    def unload_model_click(e):
        LocalGgufClient.unload_cached_model()
        refresh_models_list()

    unload_btn = ft.TextButton(
        "Clear RAM",
        icon=ft.Icons.CLEANING_SERVICES,
        icon_color=ft.Colors.AMBER_400,
        tooltip="Unload cached GGUF model from RAM/VRAM",
        on_click=unload_model_click,
        visible=False
    )

    accel_row = (
        ft.Row([
            ft.Icon(ft.Icons.BOLT, size=15, color=ft.Colors.GREEN_400),
            ft.Text(f"Hardware: {accel_info['active_backend']}", size=11, color=ft.Colors.GREEN_400, weight=ft.FontWeight.W_500)
        ], spacing=4)
        if accel_info["gpu_offload_supported"]
        else ft.Row([
            ft.Icon(ft.Icons.MEMORY, size=14, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text(f"Hardware: {accel_info['active_backend']}", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        ], spacing=4)
    )

    hardware_banner = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.COMPUTER, size=24, color=ft.Colors.PRIMARY),
            ft.Column([
                ft.Text(f"System Memory: Available {mem_info['available_gb']:.1f} GB / Total {mem_info['total_gb']:.1f} GB", size=12, weight=ft.FontWeight.BOLD),
                ft.Text(f"CPU: {cpu_info['physical_cores']} Cores (Recommended Threads: {cpu_info['optimal_threads']})", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                accel_row
            ], spacing=3, expand=True),
            unload_btn
        ], spacing=12),
        padding=10,
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border_radius=8
    )

    def refresh_models_list():
        models_listview.controls.clear()
        model_controls_map.clear()
        all_models = local_models_catalog.get_all_models()

        generation_models = [m for m in all_models if m.model_type == "generation"]
        embedding_models = [m for m in all_models if m.model_type == "embedding"]
        reranker_models = [m for m in all_models if m.model_type == "reranker"]

        groups = [
            ("Text Generation Models", generation_models),
            ("Semantic Memory & Linking Models", embedding_models),
            ("Cross-Encoder Reranker Models", reranker_models),
        ]

        for group_title, group_models in groups:
            if not group_models:
                continue

            models_listview.controls.append(
                ft.Container(
                    content=ft.Text(group_title, size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                    padding=ft.Padding.only(top=8, bottom=4)
                )
            )

            for model in group_models:
                is_downloaded = ModelDownloader.is_model_downloaded(model, models_dir)
                is_downloading = ModelDownloader.is_downloading(model.id)
                current_status = ModelDownloader.get_status(model.id)
                is_active = (model.id == active_model_id and model.model_type == "generation")

                is_runnable, hw_msg, hw_status = HardwareChecker.check_model_compatibility(model)

                if hw_status == "OK":
                    hw_badge = ft.Container(
                        content=ft.Text("🟢 System Compatible", size=11, color=ft.Colors.GREEN_400, weight=ft.FontWeight.BOLD),
                        bgcolor=ft.Colors.GREEN_900 if dm.page.theme_mode == ft.ThemeMode.DARK else ft.Colors.GREEN_100,
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                        border_radius=6
                    )
                elif hw_status == "WARNING":
                    hw_badge = ft.Container(
                        content=ft.Text("🟡 Low Memory", size=11, color=ft.Colors.AMBER_400, weight=ft.FontWeight.BOLD),
                        bgcolor=ft.Colors.AMBER_900 if dm.page.theme_mode == ft.ThemeMode.DARK else ft.Colors.AMBER_100,
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                        border_radius=6
                    )
                else:
                    hw_badge = ft.Container(
                        content=ft.Text("🔴 Insufficient Memory", size=11, color=ft.Colors.RED_400, weight=ft.FontWeight.BOLD),
                        bgcolor=ft.Colors.RED_900 if dm.page.theme_mode == ft.ThemeMode.DARK else ft.Colors.RED_100,
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                        border_radius=6
                    )

                init_progress = current_status.progress if is_downloading else 0.0
                progress_bar = ft.ProgressBar(
                    visible=is_downloading,
                    value=init_progress,
                    color=ft.Colors.PRIMARY
                )

                default_stat = f"Size: ~{model.size_gb:.1f} GB  |  Min. RAM: {model.min_ram_gb:.0f} GB"
                status_text = ft.Text(
                    (current_status.status_text or default_stat) if is_downloading else default_stat,
                    size=11,
                    color=ft.Colors.ON_SURFACE_VARIANT
                )

                action_row = ft.Row(spacing=8, alignment=ft.MainAxisAlignment.END)

                if is_downloading:
                    action_row.controls.append(
                        ft.IconButton(
                            icon=ft.Icons.CANCEL,
                            icon_color=ft.Colors.ERROR,
                            on_click=lambda e, m=model: (
                                ModelDownloader.cancel_download(m.id),
                                refresh_models_list()
                            )
                        )
                    )
                elif is_downloaded:
                    if model.model_type in ("embedding", "reranker"):
                        action_row.controls.append(
                            ft.Container(
                                content=ft.Row([
                                    ft.Icon(ft.Icons.CHECK_CIRCLE, size=16, color=ft.Colors.GREEN_400),
                                    ft.Text("Installed", size=12, color=ft.Colors.GREEN_400, weight=ft.FontWeight.BOLD)
                                ], spacing=4),
                                bgcolor=ft.Colors.GREEN_900 if dm.page.theme_mode == ft.ThemeMode.DARK else ft.Colors.GREEN_100,
                                padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                                border_radius=6
                            )
                        )
                    elif is_active:
                        action_row.controls.append(
                            ft.Container(
                                content=ft.Row([
                                    ft.Icon(ft.Icons.CHECK_CIRCLE, size=16, color=ft.Colors.PRIMARY),
                                    ft.Text("Active Model", size=12, color=ft.Colors.PRIMARY, weight=ft.FontWeight.BOLD)
                                ], spacing=4),
                                bgcolor=ft.Colors.PRIMARY_CONTAINER,
                                padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                                border_radius=6
                            )
                        )
                    else:
                        action_row.controls.append(
                            ft.Button(
                                "Select",
                                icon=ft.Icons.CHECK,
                                on_click=lambda e, m=model: (
                                    on_select_model(m.id),
                                    refresh_models_list()
                                )
                            )
                        )

                    def make_delete_handler(m_item):
                        return lambda e: (
                            LocalGgufClient.unload_cached_model(),
                            ModelDownloader.delete_model(m_item, models_dir),
                            on_model_deleted(m_item.id) if (on_model_deleted and m_item.model_type == "generation") else None,
                            refresh_models_list()
                        )

                    action_row.controls.append(
                        ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color=ft.Colors.ERROR,
                            tooltip="Delete Model File",
                            on_click=make_delete_handler(model)
                        )
                    )
                else:
                    def make_dl_handler(m_item):
                        return lambda e: (
                            ModelDownloader.start_download(
                                m_item,
                                models_dir,
                                on_finished=lambda p: refresh_models_list(),
                                on_error=lambda err: refresh_models_list()
                            ),
                            refresh_models_list()
                        )

                    dl_btn_label = "Retry" if current_status.state == "error" else "Download Model"
                    action_row.controls.append(
                        ft.Button(
                            dl_btn_label,
                            icon=ft.Icons.REFRESH if current_status.state == "error" else ft.Icons.DOWNLOAD,
                            disabled=not is_runnable,
                            tooltip=hw_msg if not is_runnable else None,
                            on_click=make_dl_handler(model)
                        )
                    )

                # Store controls for real-time polling updates
                model_controls_map[model.id] = (progress_bar, status_text)

                card = ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Text(model.display_name, size=15, weight=ft.FontWeight.BOLD),
                            hw_badge
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Text(model.user_description, size=12),
                        status_text,
                        progress_bar,
                        action_row
                    ], spacing=6),
                    padding=12,
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH if is_active else ft.Colors.SURFACE_CONTAINER,
                    border=ft.Border.all(1.5, ft.Colors.PRIMARY) if is_active else None,
                    border_radius=10
                )
                models_listview.controls.append(card)

        try:
            unload_btn.visible = LocalGgufClient.is_model_loaded()
            hardware_banner.update()
            models_listview.update()
        except Exception:
            pass

    # Real-time event loop polling task
    async def _realtime_download_poller():
        nonlocal is_dialog_active
        while is_dialog_active:
            try:
                for model in local_models_catalog.get_all_models():
                    status = ModelDownloader.get_status(model.id)
                    if model.id in model_controls_map:
                        pb, st = model_controls_map[model.id]
                        if status.state == "downloading":
                            changed = False
                            if not pb.visible:
                                pb.visible = True
                                changed = True
                            new_prog = max(0.0, min(1.0, status.progress))
                            if abs(pb.value - new_prog) > 0.001:
                                pb.value = new_prog
                                changed = True
                            if st.value != status.status_text and status.status_text:
                                st.value = status.status_text
                                st.color = ft.Colors.PRIMARY
                                st.weight = ft.FontWeight.W_500
                                changed = True
                            if changed:
                                try:
                                    pb.update()
                                    st.update()
                                except Exception:
                                    pass
                        elif status.state == "completed":
                            if pb.visible:
                                pb.visible = False
                                refresh_models_list()
                                return
                        elif status.state == "error":
                            if pb.visible:
                                pb.visible = False
                                st.value = f"Error: {status.error_message or status.status_text}"
                                st.color = ft.Colors.ERROR
                                try:
                                    pb.update()
                                    st.update()
                                except Exception:
                                    pass
            except Exception:
                pass
            await asyncio.sleep(0.2)

    def close_model_manager(e):
        nonlocal is_dialog_active
        is_dialog_active = False
        dm.close(dm.model_manager_dialog)

    dm.model_manager_dialog.content = ft.Column([
        hardware_banner,
        ft.Divider(height=1),
        models_listview
    ], tight=True, spacing=10, width=540)

    dm.model_manager_dialog.actions = [
        ft.TextButton("Close", on_click=close_model_manager)
    ]

    dm._safe_open(dm.model_manager_dialog)
    refresh_models_list()

    # Launch background poller on Flet page async event loop
    if hasattr(dm.page, "run_task"):
        try:
            dm.page.run_task(_realtime_download_poller)
        except Exception:
            pass
