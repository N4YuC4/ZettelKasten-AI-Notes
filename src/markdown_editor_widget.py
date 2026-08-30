# markdown_editor_widget.py
#
# Zettelkasten AI Notes için Markdown Editör Bileşeni.
# İki temel mod sunar:
# 1. Kaynak Modu: Markdown metninin doğrudan yazıldığı ve biçimlendirildiği editör.
# 2. Okuma Modu: Markdown metninin renderlandığı ve WikiLink bağlantılarının tıklanabildiği görünüm.

import flet as ft
import urllib.parse
from typing import Callable, Optional, List, Tuple
from note_service import (
    preserve_single_linebreaks,
    process_markdown_wikilinks,
    calculate_document_stats,
    toggle_task_in_text,
)


def _safe_update(ctrl: ft.Control):
    try:
        if ctrl.page is not None:
            ctrl.update()
    except Exception:
        pass


def _get_page(ctrl: ft.Control) -> Optional[ft.Page]:
    try:
        return ctrl.page
    except Exception:
        return None


class MarkdownEditorWidget(ft.Container):
    """
    Kaynak ve Okuma modlarını içeren Markdown Editör Bileşeni.
    """
    def __init__(
        self,
        on_content_change: Optional[Callable[[str], None]] = None,
        on_wikilink_clicked: Optional[Callable[[str], None]] = None,
        get_all_notes_callback: Optional[Callable[[], List[Tuple[str, str, str]]]] = None,
        on_save_shortcut: Optional[Callable[[], None]] = None,
        on_blur: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.on_content_change = on_content_change
        self.on_wikilink_clicked = on_wikilink_clicked
        self.get_all_notes_callback = get_all_notes_callback
        self.on_save_shortcut = on_save_shortcut
        self.on_blur = on_blur

        # Modlar: 'source' (Kaynak / Düzenleme) veya 'reading' (Okuma / Önizleme)
        self.current_mode = "source"
        self._raw_content = ""
        self._selection_start = 0
        self._selection_end = 0

        self.expand = True
        self.bgcolor = ft.Colors.TRANSPARENT

        self._init_controls()
        self._build_layout()

    def _init_controls(self):
        # Kaynak Modu: Standart metin alanı
        self.editor_field = ft.TextField(
            multiline=True,
            expand=True,
            hint_text="Notunuzu buraya yazın...\nBiçimlendirme için araç çubuğunu veya Markdown formatını (#, **, [[Not]]) kullanabilirsiniz.",
            border=ft.InputBorder.NONE,
            text_size=15,
            content_padding=ft.Padding.all(18),
            cursor_color=ft.Colors.PRIMARY,
            cursor_width=2,
            on_change=self._handle_editor_change,
            on_selection_change=self._handle_selection_change,
            on_blur=self._handle_editor_blur,
        )

        # Okuma Modu: Renderlanmış Markdown görünümü
        self.reading_markdown_view = ft.Markdown(
            value="*Henüz bir içerik yok.*",
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            code_theme="atom-one-dark",
            expand=True,
            on_tap_link=self._handle_link_tap,
        )

        self.reading_scroll_view = ft.Column(
            controls=[
                ft.Container(
                    content=self.reading_markdown_view,
                    padding=ft.Padding.all(18),
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        # Mod Seçici: Kaynak | Okuma
        self.mode_toggle_btn = ft.SegmentedButton(
            selected=["source"],
            allow_multiple_selection=False,
            show_selected_icon=True,
            segments=[
                ft.Segment(value="source", label=ft.Text("Kaynak", size=12), icon=ft.Icon(ft.Icons.EDIT_NOTE, size=16)),
                ft.Segment(value="reading", label=ft.Text("Okuma", size=12), icon=ft.Icon(ft.Icons.MENU_BOOK, size=16)),
            ],
            on_change=self._handle_mode_change,
        )

        # İstatistikler
        self.stats_text = ft.Text(
            "0 kelime | 0 karakter | 0 satır | ~0 dk",
            size=11,
            color=ft.Colors.ON_SURFACE_VARIANT,
        )

        # Kaydedilmemiş değişiklik uyarısı
        self.dirty_indicator = ft.Text(
            "",
            size=11,
            color=ft.Colors.AMBER_400,
            weight=ft.FontWeight.BOLD,
        )

        # WikiLink Seçim Modalı
        self.wikilink_dialog = ft.AlertDialog(
            title=ft.Text("WikiLink Ekle [[...]]", size=16, weight=ft.FontWeight.BOLD),
            content=ft.Container(width=350, height=300),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END,
        )

    def _create_toolbar_button(self, icon, tooltip: str, on_click: Callable):
        return ft.IconButton(
            icon=icon,
            tooltip=tooltip,
            icon_size=18,
            style=ft.ButtonStyle(
                padding=ft.Padding.all(4),
                shape=ft.RoundedRectangleBorder(radius=6),
            ),
            on_click=on_click,
        )

    def _build_toolbar(self) -> ft.Row:
        return ft.Row(
            controls=[
                # Başlıklar
                ft.PopupMenuButton(
                    icon=ft.Icons.TITLE,
                    tooltip="Başlık Seviyesi",
                    items=[
                        ft.PopupMenuItem(content=ft.Text("H1 - Başlık 1"), on_click=lambda e: self.format_heading(1)),
                        ft.PopupMenuItem(content=ft.Text("H2 - Başlık 2"), on_click=lambda e: self.format_heading(2)),
                        ft.PopupMenuItem(content=ft.Text("H3 - Başlık 3"), on_click=lambda e: self.format_heading(3)),
                        ft.PopupMenuItem(content=ft.Text("H4 - Başlık 4"), on_click=lambda e: self.format_heading(4)),
                    ],
                ),
                ft.VerticalDivider(width=1, thickness=1, color=ft.Colors.OUTLINE_VARIANT),
                # Metin Biçimleri
                self._create_toolbar_button(ft.Icons.FORMAT_BOLD, "Kalın (Ctrl+B)", lambda e: self.wrap_selection("**", "**", "kalın metin")),
                self._create_toolbar_button(ft.Icons.FORMAT_ITALIC, "İtalik (Ctrl+I)", lambda e: self.wrap_selection("*", "*", "italik metin")),
                self._create_toolbar_button(ft.Icons.FORMAT_STRIKETHROUGH, "Üstü Çizili", lambda e: self.wrap_selection("~~", "~~", "üstü çizili")),
                self._create_toolbar_button(ft.Icons.CODE, "Satır İçi Kod", lambda e: self.wrap_selection("`", "`", "kod")),
                ft.VerticalDivider(width=1, thickness=1, color=ft.Colors.OUTLINE_VARIANT),
                # Blok Biçimleri
                self._create_toolbar_button(ft.Icons.FORMAT_QUOTE, "Alıntı Bloğu", lambda e: self.prepend_lines("> ")),
                self._create_toolbar_button(ft.Icons.INTEGRATION_INSTRUCTIONS, "Kod Bloğu", lambda e: self.insert_code_block()),
                self._create_toolbar_button(ft.Icons.FORMAT_LIST_BULLETED, "Madde İmi Listesi", lambda e: self.prepend_lines("- ")),
                self._create_toolbar_button(ft.Icons.FORMAT_LIST_NUMBERED, "Numaralı Liste", lambda e: self.prepend_numbered_list()),
                self._create_toolbar_button(ft.Icons.CHECK_BOX_OUTLINED, "Görev Kutusu", lambda e: self.prepend_lines("- [ ] ")),
                ft.VerticalDivider(width=1, thickness=1, color=ft.Colors.OUTLINE_VARIANT),
                # Bağlantılar ve Tablo
                self._create_toolbar_button(ft.Icons.LINK, "Web Bağlantısı", lambda e: self.insert_link()),
                self._create_toolbar_button(ft.Icons.POLYMER, "WikiLink [[...]]", lambda e: self.open_wikilink_picker()),
                self._create_toolbar_button(ft.Icons.TABLE_CHART, "Tablo Ekle", lambda e: self.insert_table_template()),
                self._create_toolbar_button(ft.Icons.HORIZONTAL_RULE, "Yatay Çizgi", lambda e: self.insert_text("\n\n---\n\n")),
            ],
            scroll=ft.ScrollMode.AUTO,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=2,
        )

    def _build_layout(self):
        self.toolbar = self._build_toolbar()
        
        self.toolbar_container = ft.Container(
            content=self.toolbar,
            expand=True,
            padding=ft.Padding.only(left=5, right=5),
        )

        self.header_bar = ft.Container(
            content=ft.Row(
                controls=[
                    self.toolbar_container,
                    self.mode_toggle_btn,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=10, vertical=6),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=8,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

        self.main_view_area = ft.Container(
            content=self.editor_field,
            expand=True,
            padding=0,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
            border_radius=10,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            alignment=ft.Alignment(-1, -1),
        )

        self.status_bar = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Row([
                        self.dirty_indicator,
                        self.stats_text,
                    ], spacing=10),
                    ft.Text("Markdown Editör • Ctrl+E ile Mod Değiştir", size=11, color=ft.Colors.OUTLINE),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=12, vertical=4),
        )

        self.content = ft.Column(
            controls=[
                self.header_bar,
                self.main_view_area,
                self.status_bar,
            ],
            expand=True,
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    # --- Mod Yönetimi ---

    def set_mode(self, mode: str):
        """Görünüm modunu değiştirir: 'source' (Kaynak) veya 'reading' (Okuma)."""
        if mode not in ("source", "reading"):
            return
        
        self.current_mode = mode
        self.mode_toggle_btn.selected = [mode]
        
        if mode == "source":
            self.editor_field.value = self._raw_content
            self.main_view_area.content = self.editor_field
            self.toolbar.visible = True
        elif mode == "reading":
            self.reading_markdown_view.value = process_markdown_wikilinks(self._raw_content) if self._raw_content.strip() else "*Henüz içerik yok.*"
            self.main_view_area.content = self.reading_scroll_view
            self.toolbar.visible = False

        _safe_update(self)

    def toggle_mode(self):
        """Kaynak ve Okuma modları arasında geçiş yapar (Ctrl+E kısayolu)."""
        new_mode = "reading" if self.current_mode == "source" else "source"
        self.set_mode(new_mode)

    def _handle_mode_change(self, e):
        selected_items = e.control.selected
        if selected_items:
            new_mode = selected_items[0] if isinstance(selected_items, (list, tuple)) else next(iter(selected_items))
            self.set_mode(new_mode)

    # --- İçerik Yönetimi ---

    def get_value(self) -> str:
        return self._raw_content

    def set_value(self, text: str, mark_dirty: bool = False):
        self._raw_content = text or ""
        self.editor_field.value = self._raw_content
        self._selection_start = 0
        self._selection_end = 0
        self._update_stats()
        
        if self.current_mode == "reading":
            self.reading_markdown_view.value = process_markdown_wikilinks(self._raw_content) if self._raw_content.strip() else "*Henüz içerik yok.*"
            _safe_update(self.reading_markdown_view)
        else:
            _safe_update(self.editor_field)

        self.set_dirty(mark_dirty)
        _safe_update(self)

    def set_dirty(self, is_dirty: bool):
        self.dirty_indicator.value = "● Kaydedilmemiş değişiklikler" if is_dirty else ""
        _safe_update(self.dirty_indicator)

    def _handle_editor_change(self, e):
        self._raw_content = self.editor_field.value or ""
        self._update_stats()
        self.set_dirty(True)
        if self.on_content_change:
            self.on_content_change(self._raw_content)

    def _handle_editor_blur(self, e):
        if self.on_blur:
            self.on_blur()

    def _handle_selection_change(self, e):
        ctrl = e.control
        if hasattr(ctrl, 'selection') and ctrl.selection:
            sel = ctrl.selection
            self._selection_start = getattr(sel, 'start', getattr(sel, 'base_offset', 0))
            self._selection_end = getattr(sel, 'end', getattr(sel, 'extent_offset', 0))

    def _update_stats(self):
        stats = calculate_document_stats(self._raw_content)
        self.stats_text.value = f"{stats.words} kelime | {stats.chars} karakter | {stats.lines} satır | ~{stats.reading_time_min} dk okuma"
        _safe_update(self.stats_text)

    def _handle_link_tap(self, e):
        link_url = e.data
        if link_url.startswith("zettel://note/"):
            encoded_title = link_url[len("zettel://note/"):]
            note_title = urllib.parse.unquote(encoded_title)
            if self.on_wikilink_clicked:
                self.on_wikilink_clicked(note_title)
        else:
            pg = _get_page(self)
            if pg:
                pg.launch_url(link_url)

    # --- Biçimlendirme Araçları ---

    def _apply_content_update(self, new_text: str):
        self._raw_content = new_text
        self.editor_field.value = new_text
        self._update_stats()
        self.set_dirty(True)

        if self.current_mode == "source":
            _safe_update(self.editor_field)
        elif self.current_mode == "reading":
            self.reading_markdown_view.value = process_markdown_wikilinks(self._raw_content)
            _safe_update(self.reading_markdown_view)

        if self.on_content_change:
            self.on_content_change(self._raw_content)

    def insert_text(self, text_to_insert: str):
        current_text = self._raw_content
        start = self._selection_start
        end = self._selection_end
        
        if start > end:
            start, end = end, start

        start = max(0, min(start, len(current_text)))
        end = max(0, min(end, len(current_text)))

        new_text = current_text[:start] + text_to_insert + current_text[end:]
        self._selection_start = start + len(text_to_insert)
        self._selection_end = self._selection_start
        self._apply_content_update(new_text)

    def wrap_selection(self, prefix: str, suffix: str, default_text: str = ""):
        current_text = self._raw_content
        start = self._selection_start
        end = self._selection_end

        if start > end:
            start, end = end, start

        start = max(0, min(start, len(current_text)))
        end = max(0, min(end, len(current_text)))

        selected_text = current_text[start:end]
        if not selected_text:
            selected_text = default_text

        wrapped = f"{prefix}{selected_text}{suffix}"
        new_text = current_text[:start] + wrapped + current_text[end:]
        self._selection_start = start + len(prefix)
        self._selection_end = start + len(prefix) + len(selected_text)
        self._apply_content_update(new_text)

    def prepend_lines(self, prefix: str):
        current_text = self._raw_content
        start = self._selection_start
        end = self._selection_end

        if start > end:
            start, end = end, start

        line_start = current_text.rfind('\n', 0, start)
        line_start = 0 if line_start == -1 else line_start + 1

        line_end = current_text.find('\n', end)
        line_end = len(current_text) if line_end == -1 else line_end

        selected_chunk = current_text[line_start:line_end]
        lines = selected_chunk.split('\n')
        modified_lines = [f"{prefix}{line}" for line in lines]
        modified_chunk = '\n'.join(modified_lines)

        new_text = current_text[:line_start] + modified_chunk + current_text[line_end:]
        self._apply_content_update(new_text)

    def format_heading(self, level: int):
        level_str = "#" * max(1, min(level, 6)) + " "
        self.prepend_lines(level_str)

    def prepend_numbered_list(self):
        current_text = self._raw_content
        start = self._selection_start
        end = self._selection_end

        if start > end:
            start, end = end, start

        line_start = current_text.rfind('\n', 0, start)
        line_start = 0 if line_start == -1 else line_start + 1

        line_end = current_text.find('\n', end)
        line_end = len(current_text) if line_end == -1 else line_end

        selected_chunk = current_text[line_start:line_end]
        lines = selected_chunk.split('\n')
        modified_lines = [f"{i + 1}. {line}" for i, line in enumerate(lines)]
        modified_chunk = '\n'.join(modified_lines)

        new_text = current_text[:line_start] + modified_chunk + current_text[line_end:]
        self._apply_content_update(new_text)

    def insert_code_block(self):
        code_block = "\n```python\nprint('Hello, Zettelkasten!')\n```\n"
        self.insert_text(code_block)

    def insert_table_template(self):
        table_md = (
            "\n| Başlık 1 | Başlık 2 | Başlık 3 |\n"
            "| :--- | :--- | :--- |\n"
            "| Değer A | Değer B | Değer C |\n"
            "| Değer D | Değer E | Değer F |\n\n"
        )
        self.insert_text(table_md)

    def insert_link(self):
        current_text = self._raw_content
        start = self._selection_start
        end = self._selection_end
        if start > end:
            start, end = end, start
        selected = current_text[start:end] if start != end else ""
        title = selected if selected else "Bağlantı Metni"
        self.insert_text(f"[{title}](https://example.com)")

    # --- WikiLink Seçici ---

    def open_wikilink_picker(self):
        pg = _get_page(self)
        if not pg:
            return

        all_notes = []
        if self.get_all_notes_callback:
            all_notes = self.get_all_notes_callback()

        search_box = ft.TextField(
            hint_text="Notlarda ara...",
            prefix_icon=ft.Icons.SEARCH,
            dense=True,
            autofocus=True,
        )

        notes_list = ft.ListView(
            expand=True,
            spacing=4,
        )

        def select_note(title):
            self.insert_text(f"[[{title}]]")
            self.wikilink_dialog.open = False
            _safe_update(pg)

        def update_filtered_notes(query=""):
            notes_list.controls.clear()
            q = query.lower().strip()
            
            matched = 0
            for item in all_notes:
                note_id, title, cat = item
                if not q or q in title.lower() or (cat and q in cat.lower()):
                    matched += 1
                    notes_list.controls.append(
                        ft.ListTile(
                            leading=ft.Icon(ft.Icons.NOTE, size=18, color=ft.Colors.PRIMARY),
                            title=ft.Text(title, size=13, weight=ft.FontWeight.W_500),
                            subtitle=ft.Text(f"Kategori: {cat}" if cat else "Genel", size=11, color=ft.Colors.OUTLINE),
                            dense=True,
                            on_click=lambda e, t=title: select_note(t),
                        )
                    )

            if matched == 0:
                if q:
                    notes_list.controls.append(
                        ft.ListTile(
                            leading=ft.Icon(ft.Icons.ADD_LINK, size=18, color=ft.Colors.TERTIARY),
                            title=ft.Text(f"Yeni Not Bağlantısı Ekle: [[{query.strip()}]]", size=13, color=ft.Colors.TERTIARY),
                            dense=True,
                            on_click=lambda e, t=query.strip(): select_note(t),
                        )
                    )
                else:
                    notes_list.controls.append(
                        ft.Text("Kayıtlı not bulunamadı.", size=12, italic=True, color=ft.Colors.OUTLINE)
                    )
            
            _safe_update(notes_list)

        search_box.on_change = lambda e: update_filtered_notes(search_box.value)

        self.wikilink_dialog.content = ft.Container(
            content=ft.Column(
                controls=[
                    search_box,
                    ft.Divider(height=1),
                    notes_list,
                ],
                expand=True,
                spacing=8,
            ),
            width=400,
            height=350,
        )

        self.wikilink_dialog.actions = [
            ft.TextButton("İptal", on_click=lambda e: self._close_wikilink_dialog()),
        ]

        if self.wikilink_dialog not in pg.overlay:
            pg.overlay.append(self.wikilink_dialog)

        self.wikilink_dialog.open = True
        _safe_update(pg)
        update_filtered_notes("")

    def _close_wikilink_dialog(self):
        self.wikilink_dialog.open = False
        pg = _get_page(self)
        if pg:
            _safe_update(pg)
