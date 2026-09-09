# ui/dialog_manager.py
#
# Encapsulates all modal dialog creations and display workflows.

import os
import asyncio
import flet as ft
from typing import Callable, Optional, List, Tuple, Any, Dict
import local_models_catalog
from hardware_checker import HardwareChecker
from model_downloader import ModelDownloader
from local_gguf_client import LocalGgufClient


class DialogManager:
    """
    Manages all modal dialog lifecycle and event dispatching for Zettelkasten AI Notes.
    """
    def __init__(self, page: ft.Page):
        self.page = page
        self._init_dialogs()

    def _init_dialogs(self):
        # 1. Simple Action Confirm Dialogs
        self.delete_note_dialog = ft.AlertDialog(
            title=ft.Text("Delete Note"),
            content=ft.Text(""),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.delete_cat_dialog = ft.AlertDialog(
            title=ft.Text("Kategoriyi Sil"),
            content=ft.Text("Bu kategoriyi ve içerdiği tüm notları silmek istediğinizden emin misiniz? Bu işlem geri alınamaz."),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.unlink_dialog = ft.AlertDialog(
            title=ft.Text("Unlink Note"),
            content=ft.Text(""),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.error_dialog = ft.AlertDialog(
            title=ft.Text("Error"),
            content=ft.Text(""),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.unsaved_dialog = ft.AlertDialog(
            title=ft.Text("Unsaved Changes"),
            content=ft.Text("You have unsaved changes. Do you want to save them?"),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.create_linked_dialog = ft.AlertDialog(
            title=ft.Text("Yeni Not Bağlantısı"),
            content=ft.Text(""),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 2. Input Dialogs
        self.new_category_field = ft.TextField(label="Category Name", border_radius=8)
        self.new_category_dialog = ft.AlertDialog(
            title=ft.Text("New Category"),
            content=self.new_category_field,
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        self.rename_note_field = ft.TextField(label="New Note Title", border_radius=8)
        self.rename_dialog = ft.AlertDialog(
            title=ft.Text("Rename Note"),
            content=self.rename_note_field,
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 3. Complex Selection Dialogs (Link Note)
        self.link_search_field = ft.TextField(
            hint_text="Search notes to link...",
            prefix_icon=ft.Icons.SEARCH,
            border_radius=8
        )
        self.linkable_notes_listview = ft.ListView(spacing=5, height=300)
        self.link_note_dialog = ft.AlertDialog(
            title=ft.Text("Select Note to Link"),
            content=ft.Column([
                self.link_search_field,
                self.linkable_notes_listview
            ], tight=True, spacing=15, width=400),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 4. Progress / Loading Dialog
        self.loading_msg = ft.Text("Metin ayıklanıyor... Lütfen bekleyiniz.", size=13)
        self.loading_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Yapay Zeka ile Not Üretiliyor", weight=ft.FontWeight.BOLD),
            content=ft.Row([
                ft.ProgressRing(width=28, height=28, stroke_width=3),
                self.loading_msg
            ], spacing=15, alignment=ft.MainAxisAlignment.CENTER),
            actions=[]
        )

        # 5. Settings Dialog
        self.api_key_field = ft.TextField(
            label="Gemini API Key",
            password=True,
            can_reveal_password=True,
            border_radius=8
        )
        self.settings_dialog = ft.AlertDialog(
            title=ft.Text("Ayarlar"),
            content=ft.Container(width=450),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 6. Model Manager Dialog
        self.model_manager_dialog = ft.AlertDialog(
            title=ft.Row([
                ft.Icon(ft.Icons.MEMORY, color=ft.Colors.PRIMARY),
                ft.Text("Lokal Model Yöneticisi", weight=ft.FontWeight.BOLD),
            ], spacing=10),
            content=ft.Container(width=540),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # Register all dialogs in page overlay with on_dismiss sync
        dialogs = [
            self.delete_note_dialog,
            self.delete_cat_dialog,
            self.unlink_dialog,
            self.error_dialog,
            self.unsaved_dialog,
            self.create_linked_dialog,
            self.new_category_dialog,
            self.rename_dialog,
            self.link_note_dialog,
            self.loading_dialog,
            self.settings_dialog,
            self.model_manager_dialog,
        ]
        for dlg in dialogs:
            dlg.on_dismiss = lambda e, d=dlg: self._handle_dialog_dismiss(d)
            self.page.overlay.append(dlg)

    def _handle_dialog_dismiss(self, dlg: ft.AlertDialog) -> None:
        """Keeps dialog open state strictly synchronized when closed via backdrop or ESC."""
        dlg.open = False

    def close(self, dlg: ft.AlertDialog) -> None:
        """Closes any given modal dialog safely."""
        dlg.open = False
        try:
            self.page.update()
        except Exception:
            pass

    def show_error(self, title: str, message: str) -> None:
        """Displays an error alert dialog."""
        self.error_dialog.title = ft.Text(title)
        self.error_dialog.content = ft.Text(message)
        self.error_dialog.actions = [
            ft.TextButton("OK", on_click=lambda e: self.close(self.error_dialog))
        ]
        self.error_dialog.open = True
        self.page.update()

    def show_delete_note_confirm(self, note_title: str, on_confirm: Callable) -> None:
        """Shows note deletion confirmation dialog."""
        self.delete_note_dialog.content = ft.Text(
            f"Are you sure you want to delete '{note_title}'?\nThis action cannot be undone."
        )
        self.delete_note_dialog.actions = [
            ft.TextButton("Yes", on_click=lambda e: (self.close(self.delete_note_dialog), on_confirm())),
            ft.TextButton("No", on_click=lambda e: self.close(self.delete_note_dialog))
        ]
        self.delete_note_dialog.open = True
        self.page.update()

    def show_delete_category_confirm(self, category_name: str, on_confirm: Callable) -> None:
        """Shows category deletion confirmation dialog."""
        self.delete_cat_dialog.content = ft.Text(
            f"Bu kategoriyi ('{category_name}') ve içerdiği tüm notları silmek istediğinizden emin misiniz? Bu işlem geri alınamaz."
        )
        self.delete_cat_dialog.actions = [
            ft.TextButton("İptal", on_click=lambda e: self.close(self.delete_cat_dialog)),
            ft.TextButton("Sil", on_click=lambda e: (self.close(self.delete_cat_dialog), on_confirm()))
        ]
        self.delete_cat_dialog.open = True
        self.page.update()

    def show_unlink_confirm(self, target_title: str, on_confirm: Callable) -> None:
        """Shows unlinking confirmation dialog."""
        self.unlink_dialog.content = ft.Text(
            f"Are you sure you want to unlink '{target_title}' from the current note?"
        )
        self.unlink_dialog.actions = [
            ft.TextButton("Yes", on_click=lambda e: (self.close(self.unlink_dialog), on_confirm())),
            ft.TextButton("No", on_click=lambda e: self.close(self.unlink_dialog))
        ]
        self.unlink_dialog.open = True
        self.page.update()

    def show_unsaved_changes_prompt(self, on_save: Callable, on_discard: Callable) -> None:
        """Shows prompt when navigating away with unsaved changes."""
        self.unsaved_dialog.actions = [
            ft.TextButton("Yes", on_click=lambda e: (self.close(self.unsaved_dialog), on_save())),
            ft.TextButton("No", on_click=lambda e: (self.close(self.unsaved_dialog), on_discard())),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.unsaved_dialog))
        ]
        self.unsaved_dialog.open = True
        self.page.update()

    def show_create_linked_note_prompt(self, target_title: str, on_create: Callable) -> None:
        """Shows prompt when clicking a WikiLink to a non-existent note."""
        self.create_linked_dialog.content = ft.Text(
            f"'{target_title}' başlıklı bir not bulunamadı.\nYeni bir not olarak oluşturmak ister misiniz?"
        )
        self.create_linked_dialog.actions = [
            ft.TextButton("Oluştur", on_click=lambda e: (self.close(self.create_linked_dialog), on_create())),
            ft.TextButton("İptal", on_click=lambda e: self.close(self.create_linked_dialog))
        ]
        self.create_linked_dialog.open = True
        self.page.update()

    def show_new_category_dialog(self, on_submit: Callable[[str], None]) -> None:
        """Opens dialog to create a new category."""
        self.new_category_field.value = ""
        self.new_category_field.on_submit = lambda e: on_submit(self.new_category_field.value)
        self.new_category_dialog.actions = [
            ft.TextButton("Create", on_click=lambda e: on_submit(self.new_category_field.value)),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.new_category_dialog))
        ]
        self.new_category_dialog.open = True
        self.page.update()

    def show_rename_dialog(self, current_title: str, on_submit: Callable[[str], None]) -> None:
        """Opens dialog to rename a note."""
        self.rename_note_field.value = current_title
        self.rename_note_field.on_submit = lambda e: on_submit(self.rename_note_field.value)
        self.rename_dialog.actions = [
            ft.TextButton("Rename", on_click=lambda e: on_submit(self.rename_note_field.value)),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.rename_dialog))
        ]
        self.rename_dialog.open = True
        self.page.update()

    def show_link_picker(
        self,
        current_note_id: Optional[str],
        all_notes: List[Tuple[str, str, str]],
        on_link_selected: Callable[[str, str], None]
    ) -> None:
        """Opens search list dialog to link to another note."""
        def update_link_list(query=""):
            self.linkable_notes_listview.controls.clear()
            q = query.lower().strip()
            for nid, title, _ in all_notes:
                if nid == current_note_id:
                    continue
                if not q or q in title.lower():
                    item = ft.ListTile(
                        leading=ft.Icon(ft.Icons.ARTICLE_OUTLINED, color=ft.Colors.PRIMARY),
                        title=ft.Text(title, weight=ft.FontWeight.W_500),
                        on_click=lambda e, target_id=nid, target_title=title: (
                            self.close(self.link_note_dialog),
                            on_link_selected(target_id, target_title)
                        ),
                        hover_color=ft.Colors.ON_INVERSE_SURFACE
                    )
                    self.linkable_notes_listview.controls.append(item)
            self.linkable_notes_listview.update()

        self.link_search_field.value = ""
        self.link_search_field.on_change = lambda e: update_link_list(self.link_search_field.value)
        self.link_note_dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.link_note_dialog))
        ]
        self.link_note_dialog.open = True
        self.page.update()
        update_link_list("")

    def show_loading(
        self,
        title: str = "Yapay Zeka Not Çıkarma",
        message: str = "Hazırlanıyor...",
        on_cancel: Optional[Callable[[], None]] = None
    ) -> None:
        """Shows loading progress dialog with custom text and optional cancellation."""
        self.loading_dialog.title = ft.Text(title, weight=ft.FontWeight.BOLD)
        self.loading_msg.value = message
        self.loading_dialog.content = ft.Row([
            ft.ProgressRing(width=28, height=28, stroke_width=3),
            self.loading_msg
        ], spacing=15, alignment=ft.MainAxisAlignment.CENTER)
        if on_cancel:
            self.loading_dialog.actions = [
                ft.TextButton("İptal", on_click=lambda e: on_cancel())
            ]
        else:
            self.loading_dialog.actions = []
        self.loading_dialog.open = True
        try:
            self.page.update()
        except Exception:
            pass

    def update_loading_message(self, message: str) -> None:
        """Updates text of an already open loading dialog."""
        self.loading_msg.value = message
        try:
            self.loading_msg.update()
        except Exception:
            try:
                self.page.update()
            except Exception:
                pass

    def hide_loading(self) -> None:
        """Hides the loading dialog."""
        self.loading_dialog.open = False
        try:
            self.page.update()
        except Exception:
            pass

    def show_settings_dialog(
        self,
        theme_btn: ft.IconButton,
        auto_save_switch: ft.Switch,
        current_ai_provider: str,
        current_active_model_id: str,
        on_save_settings: Callable[[str, str, str, bool], None],
        on_open_model_manager: Callable[[], None],
        current_gpu_acceleration: bool = True
    ) -> None:
        """Opens settings dialog for theme, auto-save, AI provider, and local model config."""
        self.api_key_field.value = os.getenv("GEMINI_API_KEY", "")

        selected_provider = current_ai_provider or "gemini"
        selected_model_id = current_active_model_id or local_models_catalog.DEFAULT_MODEL_ID

        gpu_switch = ft.Switch(
            value=current_gpu_acceleration,
            tooltip="Açıkken ekran kartını (Vulkan / CUDA) kullanarak daha hızlı çıkarım yapar. Kapatıldığında sadece işlemci (CPU) kullanılır."
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
                        f"Donanım Desteği: {accel_info['active_backend']}",
                        size=11,
                        color=ft.Colors.GREEN_400 if accel_info["gpu_offload_supported"] else ft.Colors.AMBER_400,
                        weight=ft.FontWeight.W_500
                    )
                ], spacing=4)

                gpu_setting_row = ft.Row([
                    ft.Column([
                        ft.Text("GPU Donanım Hızlandırma", weight=ft.FontWeight.W_600, size=13),
                        ft.Text(
                            "Ekran kartını kullanarak not çıkarmayı hızlandırır (Vulkan / CUDA).",
                            size=11,
                            color=ft.Colors.ON_SURFACE_VARIANT
                        )
                    ], expand=True, spacing=1),
                    gpu_switch
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER)

                provider_details_container.content = ft.Column([
                    ft.Text("Lokal GGUF Model Yapılandırması", weight=ft.FontWeight.BOLD),
                    ft.Text("Model doğrudan cihazınızda çalışır, hiçbir veri dışarıya gönderilmez.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Container(
                        content=ft.Row([
                            ft.Column([
                                ft.Text(f"Aktif Model: {display_title}", weight=ft.FontWeight.W_600),
                                ft.Text(model_info.user_description if model_info else "", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                                accel_badge
                            ], expand=True, spacing=2),
                            ft.Button(
                                "Model Yöneticisi",
                                icon=ft.Icons.MEMORY,
                                on_click=lambda e: (self.close(self.settings_dialog), on_open_model_manager())
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
                    ft.Text("Gemini API Yapılandırması", weight=ft.FontWeight.BOLD),
                    ft.Text("API anahtarınız yerel .env dosyanızda güvenle saklanır.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                    self.api_key_field
                ], spacing=10)
            try:
                self.settings_dialog.update()
            except Exception:
                pass

        provider_dropdown = ft.Dropdown(
            label="Yapay Zeka Sağlayıcısı",
            value=selected_provider,
            options=[
                ft.dropdown.Option(key="gemini", text="Google Gemini (Bulut)"),
                ft.dropdown.Option(key="local", text="Lokal GGUF Model (Cihazda / Çevrimdışı)"),
            ],
            border_radius=8,
            on_select=lambda e: update_provider_view(provider_dropdown.value or "gemini")
        )

        update_provider_view(selected_provider)

        self.settings_dialog.content = ft.Column([
            ft.Row([ft.Text("Tema Modu", weight=ft.FontWeight.BOLD), theme_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Text("Otomatik Kaydet", weight=ft.FontWeight.BOLD), auto_save_switch], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(),
            provider_dropdown,
            provider_details_container
        ], tight=True, spacing=15, width=450, scroll=ft.ScrollMode.AUTO)

        self.settings_dialog.actions = [
            ft.TextButton("Kaydet", on_click=lambda e: (
                on_save_settings(
                    (self.api_key_field.value or "").strip(),
                    provider_dropdown.value or "gemini",
                    selected_model_id,
                    bool(gpu_switch.value)
                ),
                self.close(self.settings_dialog)
            )),
            ft.TextButton("İptal", on_click=lambda e: self.close(self.settings_dialog))
        ]
        self.settings_dialog.open = True
        try:
            self.page.update()
        except Exception:
            pass

    def show_model_manager_dialog(
        self,
        models_dir: str,
        active_model_id: str,
        on_select_model: Callable[[str], None],
        on_model_deleted: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Opens Model Manager dialog displaying curated models, hardware status, and download controls."""
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
            "RAM'i Temizle",
            icon=ft.Icons.CLEANING_SERVICES,
            icon_color=ft.Colors.AMBER_400,
            tooltip="Bellekteki GGUF modelini RAM/VRAM'den kaldır",
            on_click=unload_model_click,
            visible=False
        )

        accel_row = (
            ft.Row([
                ft.Icon(ft.Icons.BOLT, size=15, color=ft.Colors.GREEN_400),
                ft.Text(f"Donanım: {accel_info['active_backend']}", size=11, color=ft.Colors.GREEN_400, weight=ft.FontWeight.W_500)
            ], spacing=4)
            if accel_info["gpu_offload_supported"]
            else ft.Row([
                ft.Icon(ft.Icons.MEMORY, size=14, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(f"Donanım: {accel_info['active_backend']}", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
            ], spacing=4)
        )

        hardware_banner = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.COMPUTER, size=24, color=ft.Colors.PRIMARY),
                ft.Column([
                    ft.Text(f"Sistem Belleği: Boş {mem_info['available_gb']:.1f} GB / Toplam {mem_info['total_gb']:.1f} GB", size=12, weight=ft.FontWeight.BOLD),
                    ft.Text(f"İşlemci: {cpu_info['physical_cores']} Çekirdek (Önerilen İş Parçacığı: {cpu_info['optimal_threads']})", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
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
            curated_models = local_models_catalog.get_curated_models()

            for model in curated_models:
                is_downloaded = ModelDownloader.is_model_downloaded(model, models_dir)
                is_downloading = ModelDownloader.is_downloading(model.id)
                current_status = ModelDownloader.get_status(model.id)
                is_active = (model.id == active_model_id)

                is_runnable, hw_msg, hw_status = HardwareChecker.check_model_compatibility(model)

                if hw_status == "OK":
                    hw_badge = ft.Container(
                        content=ft.Text("🟢 Sistem Uyumlu", size=11, color=ft.Colors.GREEN_400, weight=ft.FontWeight.BOLD),
                        bgcolor=ft.Colors.GREEN_900 if self.page.theme_mode == ft.ThemeMode.DARK else ft.Colors.GREEN_100,
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                        border_radius=6
                    )
                elif hw_status == "WARNING":
                    hw_badge = ft.Container(
                        content=ft.Text("🟡 Sınırlı Bellek", size=11, color=ft.Colors.AMBER_400, weight=ft.FontWeight.BOLD),
                        bgcolor=ft.Colors.AMBER_900 if self.page.theme_mode == ft.ThemeMode.DARK else ft.Colors.AMBER_100,
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                        border_radius=6
                    )
                else:
                    hw_badge = ft.Container(
                        content=ft.Text("🔴 Yetersiz Bellek", size=11, color=ft.Colors.RED_400, weight=ft.FontWeight.BOLD),
                        bgcolor=ft.Colors.RED_900 if self.page.theme_mode == ft.ThemeMode.DARK else ft.Colors.RED_100,
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                        border_radius=6
                    )

                init_progress = current_status.progress if is_downloading else 0.0
                progress_bar = ft.ProgressBar(
                    visible=is_downloading,
                    value=init_progress,
                    color=ft.Colors.PRIMARY
                )

                default_stat = f"Boyut: ~{model.size_gb:.1f} GB  |  Min. RAM: {model.min_ram_gb:.0f} GB"
                if is_downloading and current_status.status_text:
                    default_stat = current_status.status_text
                elif current_status.state == "error" and current_status.error_message:
                    default_stat = f"Hata: {current_status.error_message}"

                status_text = ft.Text(
                    default_stat,
                    size=11,
                    weight=ft.FontWeight.W_500 if is_downloading else ft.FontWeight.NORMAL,
                    color=ft.Colors.ERROR if current_status.state == "error" else (ft.Colors.PRIMARY if is_downloading else ft.Colors.ON_SURFACE_VARIANT)
                )

                action_row = ft.Row(spacing=8, alignment=ft.MainAxisAlignment.END)

                if is_downloading:
                    action_row.controls.append(
                        ft.TextButton(
                            "İptal",
                            icon=ft.Icons.CANCEL,
                            icon_color=ft.Colors.ERROR,
                            on_click=lambda e, m=model: (
                                ModelDownloader.cancel_download(m.id),
                                refresh_models_list()
                            )
                        )
                    )
                elif is_downloaded:
                    if is_active:
                        action_row.controls.append(
                            ft.Container(
                                content=ft.Row([
                                    ft.Icon(ft.Icons.CHECK_CIRCLE, size=16, color=ft.Colors.PRIMARY),
                                    ft.Text("Aktif Model", size=12, color=ft.Colors.PRIMARY, weight=ft.FontWeight.BOLD)
                                ], spacing=4),
                                bgcolor=ft.Colors.PRIMARY_CONTAINER,
                                padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                                border_radius=6
                            )
                        )
                    else:
                        action_row.controls.append(
                            ft.Button(
                                "Seç",
                                icon=ft.Icons.CHECK,
                                on_click=lambda e, m=model: (
                                    on_select_model(m.id),
                                    refresh_models_list()
                                )
                            )
                        )
                    action_row.controls.append(
                        ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color=ft.Colors.ERROR,
                            tooltip="Model Dosyasını Sil",
                            on_click=lambda e, m=model: (
                                LocalGgufClient.unload_cached_model(),
                                ModelDownloader.delete_model(m, models_dir),
                                on_model_deleted(m.id) if on_model_deleted else None,
                                refresh_models_list()
                            )
                        )
                    )
                else:
                    def start_dl_click(e, m=model):
                        ModelDownloader.start_download(
                            m,
                            models_dir,
                            on_finished=lambda p: refresh_models_list(),
                            on_error=lambda err: refresh_models_list()
                        )
                        refresh_models_list()

                    dl_btn_label = "Tekrar Dene" if current_status.state == "error" else "Modeli İndir"
                    action_row.controls.append(
                        ft.Button(
                            dl_btn_label,
                            icon=ft.Icons.REFRESH if current_status.state == "error" else ft.Icons.DOWNLOAD,
                            disabled=not is_runnable,
                            tooltip=hw_msg if not is_runnable else None,
                            on_click=start_dl_click
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
                    for model in local_models_catalog.get_curated_models():
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
                                    st.value = f"Hata: {status.error_message or status.status_text}"
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
            self.close(self.model_manager_dialog)

        self.model_manager_dialog.content = ft.Column([
            hardware_banner,
            ft.Divider(height=1),
            models_listview
        ], tight=True, spacing=10, width=540)

        self.model_manager_dialog.actions = [
            ft.TextButton("Kapat", on_click=close_model_manager)
        ]

        self.model_manager_dialog.open = True
        self.page.update()
        refresh_models_list()

        # Launch background poller on Flet page async event loop
        if hasattr(self.page, "run_task"):
            try:
                self.page.run_task(_realtime_download_poller)
            except Exception:
                pass
