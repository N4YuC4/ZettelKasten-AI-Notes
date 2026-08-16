# main.py
#
# This file is the main entry point for the Zettelkasten AI Notes application.
# It provides a highly esthetic desktop application interface using Flet and includes
# note management, real-time Markdown preview, category management, AI note creation from PDF,
# and interactive canvas-based mind mapping.

import sys
import os
import threading
import flet as ft
from dotenv import set_key
from logger import log_debug, log_error
import database_manager
import note_manager
import pdf_processor
from ai_note_generator_worker import AiNoteGeneratorWorker
from mind_map_widget import MindMapWidget

def main(page: ft.Page):
    page.title = "Zettelkasten AI Notes"
    page.theme_mode = ft.ThemeMode.DARK # Default theme mode
    page.window_width = 1400
    page.window_height = 900
    page.padding = 10
    
    # Initialize the database manager
    db_manager = database_manager.DatabaseManager()
    
    # Selected state variables
    current_note_id = None
    current_note_category = ""
    displayed_notes = [] # Stores (id, title, category) of notes currently listed
    is_dirty = False
    
    # Load and apply theme from database
    saved_theme = db_manager.get_setting("UI_THEME")
    if saved_theme:
        page.theme_mode = ft.ThemeMode.DARK if saved_theme == "Dark" else ft.ThemeMode.LIGHT
    else:
        db_manager.set_setting("UI_THEME", "Dark")
        
    # Helper to show notifications
    def show_snack_bar(message, color=ft.Colors.PRIMARY):
        page.snack_bar = ft.SnackBar(
            content=ft.Text(message, color=ft.Colors.ON_PRIMARY_CONTAINER),
            bgcolor=color,
            duration=3000
        )
        page.snack_bar.open = True
        page.update()

    # Helper to close modal dialogs safely
    def close_dialog(dlg):
        dlg.open = False
        page.update()

    # Callback when a note is selected in the Zihin Haritası
    def handle_map_note_selected(note_id):
        log_debug(f"DEBUG: Note selected from Mind Map: {note_id}")
        open_note_by_id(note_id)

    # Instantiate the Mind Map widget
    mind_map_widget = MindMapWidget(db_manager, on_note_selected=handle_map_note_selected)

    # UI CONTROLS DECLARATION
    # -----------------------
    # Left Sidebar controls
    category_dropdown = ft.Dropdown(
        label="Category",
        options=[ft.dropdown.Option("All Notes")],
        value="All Notes",
        expand=True,
        on_select=lambda e: load_notes()
    )
    
    search_textfield = ft.TextField(
        hint_text="Search notes...",
        prefix_icon=ft.Icons.SEARCH,
        border_radius=8,
        on_change=lambda e: filter_notes()
    )
    
    notes_listview = ft.ListView(
        expand=True,
        spacing=8,
        padding=5
    )
    
    note_count_label = ft.Text(
        "All Notes count: 0",
        size=12,
        color=ft.Colors.ON_SURFACE_VARIANT,
        weight=ft.FontWeight.BOLD
    )
    
    # Middle Panel Editor controls
    editor_textfield = ft.TextField(
        multiline=True,
        min_lines=20,
        max_lines=35,
        expand=True,
        hint_text="Write your notes in Markdown here...",
        border=ft.InputBorder.NONE,
        on_change=lambda e: handle_editor_change(e)
    )
    
    preview_markdown = ft.Markdown(
        value="*Markdown preview will appear here...*",
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        expand=True
    )
    
    # Right Sidebar controls
    linked_notes_listview = ft.ListView(
        expand=True,
        spacing=5,
        padding=5
    )


    # REUSABLE MODAL DIALOGS
    delete_dialog = ft.AlertDialog(title=ft.Text("Delete Note"), content=ft.Text(""), actions=[], actions_alignment=ft.MainAxisAlignment.END)
    delete_cat_dialog = ft.AlertDialog(
        title=ft.Text("Kategoriyi Sil"),
        content=ft.Text("Bu kategoriyi ve içerdiği tüm notları silmek istediğinizden emin misiniz? Bu işlem geri alınamaz."),
        actions=[],
        actions_alignment=ft.MainAxisAlignment.END
    )
    unlink_dialog = ft.AlertDialog(title=ft.Text("Unlink Note"), content=ft.Text(""), actions=[], actions_alignment=ft.MainAxisAlignment.END)
    error_dialog = ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(""), actions=[], actions_alignment=ft.MainAxisAlignment.END)
    unsaved_dialog = ft.AlertDialog(title=ft.Text("Unsaved Changes"), content=ft.Text("You have unsaved changes. Do you want to save them?"), actions=[], actions_alignment=ft.MainAxisAlignment.END)
    
    for dlg in [delete_dialog, delete_cat_dialog, unlink_dialog, error_dialog, unsaved_dialog]:
        page.overlay.append(dlg)

    # MODAL DIALOGS DEFINITIONS
    # -------------------------
    # Loading Dialog for PDF Extraction and AI Note generation
    loading_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Generating AI Notes From PDF"),
        content=ft.Row([
            ft.ProgressRing(),
            ft.Text("Extracting text from PDF... This may take a moment.")
        ], spacing=20, alignment=ft.MainAxisAlignment.CENTER)
    )
    page.overlay.append(loading_dialog)

    # API Key entry Dialog
    api_key_field = ft.TextField(
        label="Gemini API Key",
        password=True,
        can_reveal_password=True,
        border_radius=8
    )

    # Create New Category Dialog
    new_category_field = ft.TextField(
        label="Category Name",
        border_radius=8,
        on_submit=lambda e: save_new_category(e)
    )
    
    def save_new_category(e):
        category_name = new_category_field.value.strip()
        if not category_name:
            show_snack_bar("Category name cannot be empty.", color=ft.Colors.ERROR)
            return
            
        all_notes_metadata, all_categories = note_manager.load_all_notes_metadata(db_manager)
        existing_options = {opt.key for opt in category_dropdown.options if opt.key} | set(all_categories)
        if category_name in existing_options:
            show_snack_bar(f"Category '{category_name}' already exists.", color=ft.Colors.ERROR)
            return
            
        nonlocal current_note_category
        current_note_category = category_name
        
        # Reload categories with the new category selected, keeping editor content intact
        load_categories(select_category=category_name)
        load_notes(category_to_select=category_name)
        
        new_category_field.value = ""
        new_category_dialog.open = False
        page.update()
        show_snack_bar(f"Category '{category_name}' created successfully.")
        
    new_category_dialog = ft.AlertDialog(
        title=ft.Text("New Category"),
        content=new_category_field,
        actions=[
            ft.TextButton("Create", on_click=save_new_category),
            ft.TextButton("Cancel", on_click=lambda e: close_dialog(new_category_dialog))
        ],
        actions_alignment=ft.MainAxisAlignment.END
    )
    page.overlay.append(new_category_dialog)

    # Rename Note Dialog
    rename_note_field = ft.TextField(
        label="New Note Title",
        border_radius=8,
        on_submit=lambda e: save_rename_note(e)
    )
    rename_target_id = None
    
    def save_rename_note(e):
        new_title = rename_note_field.value.strip()
        if not new_title:
            show_snack_bar("Note title cannot be empty.", color=ft.Colors.ERROR)
            return
            
        nonlocal rename_target_id, current_note_id
        target_note_data = db_manager.get_note(rename_target_id)
        target_cat = target_note_data[3] if target_note_data else current_note_category

        success, new_display_title = note_manager.rename_note(
            db_manager,
            rename_target_id,
            new_title,
            target_cat
        )
        
        if success:
            if current_note_id == rename_target_id:
                page.title = f"Zettelkasten AI Notes - {new_display_title}"
                if editor_textfield.value:
                    lines = editor_textfield.value.split('\n')
                    if lines:
                        lines[0] = f"# {new_display_title}"
                    else:
                        lines = [f"# {new_display_title}"]
                    editor_textfield.value = '\n'.join(lines)
                else:
                    editor_textfield.value = f"# {new_display_title}\n\n"
                editor_textfield.update()
                update_preview()
                
            rename_dialog.open = False
            rename_note_field.value = ""
            
            if hasattr(mind_map_widget, 'invalidate_cache'):
                mind_map_widget.invalidate_cache()

            load_notes(category_to_select=current_note_category)
            display_linked_notes()
            update_mind_map()
            page.update()
            show_snack_bar("Note renamed successfully.")
        else:
            show_snack_bar(f"Failed to rename note: {new_display_title}", color=ft.Colors.ERROR)
            
    rename_dialog = ft.AlertDialog(
        title=ft.Text("Rename Note"),
        content=rename_note_field,
        actions=[
            ft.TextButton("Rename", on_click=save_rename_note),
            ft.TextButton("Cancel", on_click=lambda e: close_dialog(rename_dialog))
        ],
        actions_alignment=ft.MainAxisAlignment.END
    )
    page.overlay.append(rename_dialog)

    def rename_note_dialog(note_id, current_title):
        nonlocal rename_target_id
        rename_target_id = note_id
        rename_note_field.value = current_title
        rename_dialog.open = True
        page.update()

    # Link Note Dialog (Search and Select to Link)
    link_search_field = ft.TextField(
        hint_text="Search notes to link...",
        prefix_icon=ft.Icons.SEARCH,
        border_radius=8,
        on_change=lambda e: load_linkable_notes()
    )
    linkable_notes_listview = ft.ListView(
        spacing=5,
        height=300
    )
    
    def load_linkable_notes():
        linkable_notes_listview.controls.clear()
        search_text = link_search_field.value.lower()
        all_notes_metadata, _ = note_manager.load_all_notes_metadata(db_manager)
        
        for note_id, title, _ in all_notes_metadata:
            if note_id == current_note_id: # Cannot link to itself
                continue
                
            if search_text in title.lower():
                def make_link_handler(target_id=note_id, target_title=title):
                    return lambda e: create_link(target_id, target_title)
                    
                item = ft.ListTile(
                    leading=ft.Icon(ft.Icons.ARTICLE_OUTLINED, color=ft.Colors.PRIMARY),
                    title=ft.Text(title, weight=ft.FontWeight.W_500),
                    on_click=make_link_handler(),
                    hover_color=ft.Colors.ON_INVERSE_SURFACE
                )
                linkable_notes_listview.controls.append(item)
        linkable_notes_listview.update()
        
    def create_link(target_id, target_title):
        success = db_manager.insert_note_link(current_note_id, target_id)
        if success:
            show_snack_bar(f"Successfully linked to '{target_title}'.")
            link_note_dialog.open = False
            page.update()
            display_linked_notes()
            update_mind_map()
        else:
            show_snack_bar(f"Link to '{target_title}' already exists or failed.", color=ft.Colors.ERROR)
            
    link_note_dialog = ft.AlertDialog(
        title=ft.Text("Select Note to Link"),
        content=ft.Column([
            link_search_field,
            linkable_notes_listview
        ], tight=True, spacing=15, width=400),
        actions=[
            ft.TextButton("Cancel", on_click=lambda e: close_dialog(link_note_dialog))
        ],
        actions_alignment=ft.MainAxisAlignment.END
    )
    page.overlay.append(link_note_dialog)

    # EVENT HANDLERS
    # --------------
    def load_categories(select_category=None):
        category_dropdown.options.clear()
        category_dropdown.options.append(ft.dropdown.Option(key="", text="All Notes"))
        
        all_notes_metadata, all_categories = note_manager.load_all_notes_metadata(db_manager)
        categories_set = set(all_categories)
        if select_category and select_category.strip() and select_category != "All Notes":
            categories_set.add(select_category)
        if current_note_category and current_note_category.strip() and current_note_category != "All Notes":
            categories_set.add(current_note_category)

        for category in sorted(list(categories_set)):
            if category.strip():
                category_dropdown.options.append(ft.dropdown.Option(category))
                
        if select_category is not None:
            category_dropdown.value = select_category
        else:
            category_dropdown.value = ""
            
        category_dropdown.update()

    def load_notes(category_to_select=None):
        notes_listview.controls.clear()
        
        selected_category = category_dropdown.value or ""
        if category_to_select is not None:
            category_dropdown.value = category_to_select
            selected_category = category_to_select
            category_dropdown.update()
            
        nonlocal displayed_notes
        all_notes_metadata, _ = note_manager.load_all_notes_metadata(db_manager)
        displayed_notes = all_notes_metadata
        
        for note_id, display_title, category_path in all_notes_metadata:
            # Filter by category
            if selected_category == "" or category_path == selected_category:
                
                is_selected = (note_id == current_note_id)
                bg_color = ft.Colors.PRIMARY_CONTAINER if is_selected else ft.Colors.TRANSPARENT
                text_color = ft.Colors.ON_PRIMARY_CONTAINER if is_selected else ft.Colors.ON_SURFACE
                
                def make_click_handler(nid=note_id, title=display_title, cat=category_path):
                    return lambda e: open_note(nid, title, cat)
                    
                item = ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.ARTICLE, color=ft.Colors.PRIMARY),
                        ft.Text(display_title, color=text_color, weight=ft.FontWeight.BOLD, expand=True),
                        ft.IconButton(
                            icon=ft.Icons.EDIT,
                            icon_color=ft.Colors.ON_SURFACE_VARIANT,
                            on_click=lambda e, nid=note_id, title=display_title: rename_note_dialog(nid, title),
                            icon_size=16,
                            tooltip="Rename Note"
                        ),
                        ft.IconButton(
                            icon=ft.Icons.DELETE,
                            icon_color=ft.Colors.ERROR,
                            on_click=lambda e, nid=note_id, title=display_title: delete_note_confirm(nid, title),
                            icon_size=16,
                            tooltip="Delete Note"
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.Padding.all(8),
                    border_radius=8,
                    bgcolor=bg_color,
                    on_click=make_click_handler(),
                    on_hover=lambda e, bg=bg_color: setattr(e.control, 'bgcolor', ft.Colors.ON_INVERSE_SURFACE if e.data == "true" else bg) or e.control.update()
                )
                notes_listview.controls.append(item)
                
        notes_listview.update()
        
        # Update note count
        count = db_manager.note_count(selected_category)
        note_count_label.value = f"{selected_category} count: {count}"
        note_count_label.update()
        
        update_mind_map()

    def filter_notes():
        notes_listview.controls.clear()
        search_query = search_textfield.value.lower()
        selected_category = category_dropdown.value or ""
        
        for note_id, display_title, category_path in displayed_notes:
            if selected_category == "" or category_path == selected_category:
                if search_query in display_title.lower():
                    is_selected = (note_id == current_note_id)
                    bg_color = ft.Colors.PRIMARY_CONTAINER if is_selected else ft.Colors.TRANSPARENT
                    text_color = ft.Colors.ON_PRIMARY_CONTAINER if is_selected else ft.Colors.ON_SURFACE
                    
                    def make_click_handler(nid=note_id, title=display_title, cat=category_path):
                        return lambda e: open_note(nid, title, cat)
                        
                    item = ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.ARTICLE, color=ft.Colors.PRIMARY),
                            ft.Text(display_title, color=text_color, weight=ft.FontWeight.BOLD, expand=True),
                            ft.IconButton(
                                icon=ft.Icons.EDIT,
                                icon_color=ft.Colors.ON_SURFACE_VARIANT,
                                on_click=lambda e, nid=note_id, title=display_title: rename_note_dialog(nid, title),
                                icon_size=16
                            ),
                            ft.IconButton(
                                icon=ft.Icons.DELETE,
                                icon_color=ft.Colors.ERROR,
                                on_click=lambda e, nid=note_id, title=display_title: delete_note_confirm(nid, title),
                                icon_size=16
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        padding=ft.Padding.all(8),
                        border_radius=8,
                        bgcolor=bg_color,
                        on_click=make_click_handler(),
                        on_hover=lambda e, bg=bg_color: setattr(e.control, 'bgcolor', ft.Colors.ON_INVERSE_SURFACE if e.data == "true" else bg) or e.control.update()
                    )
                    notes_listview.controls.append(item)
        notes_listview.update()


    def try_switch_note(target_fn, *args):
        if is_dirty:
            if auto_save_switch.value:
                save_note_action(None)
                target_fn(*args)
            else:
                def yes_click(e):
                    save_note_action(None)
                    unsaved_dialog.open = False
                    page.update()
                    target_fn(*args)
                def no_click(e):
                    nonlocal is_dirty
                    is_dirty = False
                    unsaved_dialog.open = False
                    page.update()
                    target_fn(*args)
                unsaved_dialog.actions = [
                    ft.TextButton("Yes", on_click=yes_click),
                    ft.TextButton("No", on_click=no_click),
                    ft.TextButton("Cancel", on_click=lambda e: close_dialog(unsaved_dialog))
                ]
                unsaved_dialog.open = True
                page.update()
        else:
            target_fn(*args)

    def open_note_internal(note_id, display_title, category_path):
        log_debug(f"DEBUG: Opening note: {display_title} (ID: {note_id})")
        nonlocal current_note_id, current_note_category
        current_note_id = note_id
        current_note_category = category_path or ""
        
        # Update app title
        page.title = f"Zettelkasten AI Notes - {display_title}"
        
        # Load content
        content = note_manager.get_note_content(db_manager, note_id)
        if content is not None:
            editor_textfield.value = content
            editor_textfield.update()
            update_preview()
            
            # Refresh layouts
            load_notes(category_to_select=current_note_category)
            display_linked_notes()
        else:
            show_snack_bar(f"Could not read content for note: {display_title}", color=ft.Colors.ERROR)
            new_note()

    def open_note_by_id_internal(note_id):
        if not note_id:
            return
        all_notes_metadata, _ = note_manager.load_all_notes_metadata(db_manager)
        for nid, title, category_path in all_notes_metadata:
            if nid == note_id:
                open_note(nid, title, category_path)
                break

    def handle_editor_change(e):
        nonlocal is_dirty
        is_dirty = True
        update_preview()

    def update_preview():
        markdown_text = editor_textfield.value
        preview_markdown.value = markdown_text if markdown_text.strip() else "*Markdown preview will appear here...*"
        preview_markdown.update()

    def new_note_internal():
        nonlocal current_note_id, current_note_category
        editor_textfield.value = ""
        editor_textfield.update()
        update_preview()
        
        current_note_id = None
        current_note_category = category_dropdown.value if category_dropdown.value != "All Notes" else ""
        page.title = "Zettelkasten AI Notes - New Note"
        page.update()


    def open_note(note_id, display_title, category_path):
        try_switch_note(open_note_internal, note_id, display_title, category_path)

    def open_note_by_id(note_id):
        try_switch_note(open_note_by_id_internal, note_id)

    def new_note():
        try_switch_note(new_note_internal)

    def save_note_action(e):
        nonlocal current_note_id, current_note_category
        note_content = editor_textfield.value
        if not note_content.strip():
            show_snack_bar("Cannot save empty note.", color=ft.Colors.ERROR)
            return
            
        category_to_save = current_note_category if current_note_category != "All Notes" else ""
        
        note_id, display_title = note_manager.save_note(
            db_manager,
            current_note_id,
            note_content,
            category_to_save
        )
        

        if note_id and display_title:
            is_dirty = False
            current_note_id = note_id
            current_note_category = category_to_save
            page.title = f"Zettelkasten AI Notes - {display_title}"
            page.update()
            
            show_snack_bar(f"Note '{display_title}' saved successfully.")
            load_categories(select_category=category_to_save)
            load_notes(category_to_select=category_to_save)
            display_linked_notes()
        else:
            show_snack_bar("Failed to save note.", color=ft.Colors.ERROR)

    def delete_note_confirm(note_id, note_title):
        def yes_click(e):
            nonlocal current_note_id
            success = note_manager.delete_note(db_manager, note_id)
            if success:
                if current_note_id == note_id:
                    new_note()
                delete_dialog.open = False
                if hasattr(mind_map_widget, 'invalidate_cache'):
                    mind_map_widget.invalidate_cache()
                load_notes(category_to_select=current_note_category)
                display_linked_notes()
                update_mind_map()
                page.update()
                show_snack_bar(f"Note '{note_title}' deleted successfully.")
            else:
                delete_dialog.open = False
                page.update()
                show_snack_bar("Failed to delete note.", color=ft.Colors.ERROR)
                
        delete_dialog.content = ft.Text(f"Are you sure you want to delete '{note_title}'?\nThis action cannot be undone.")
        delete_dialog.actions = [
            ft.TextButton("Yes", on_click=yes_click),
            ft.TextButton("No", on_click=lambda e: close_dialog(delete_dialog))
        ]
        delete_dialog.open = True
        page.update()

    def delete_current_note(e):
        if not current_note_id:
            show_snack_bar("Please select or save a note first to delete.", color=ft.Colors.ERROR)
            return
            
        note_data = db_manager.get_note(current_note_id)
        if note_data:
            delete_note_confirm(current_note_id, note_data[1])

    def delete_category_action(e):
        selected_category = category_dropdown.value
        if not selected_category or selected_category == "All Notes":
            show_snack_bar("Please select a valid category to delete.", color=ft.Colors.ERROR)
            return
            
        def yes_click(e):
            nonlocal current_note_category, current_note_id, is_dirty
            cat_to_delete = selected_category
            success = db_manager.delete_category(cat_to_delete)
            if success:
                current_note_category = ""
                current_note_id = None
                is_dirty = False
                editor_textfield.value = ""
                editor_textfield.update()
                update_preview()
                page.title = "Zettelkasten AI Notes - New Note"
                
                category_dropdown.value = ""
                load_categories(select_category="")
                load_notes(category_to_select="")
                delete_cat_dialog.open = False
                if hasattr(mind_map_widget, 'invalidate_cache'):
                    mind_map_widget.invalidate_cache()
                display_linked_notes()
                update_mind_map()
                page.update()
                show_snack_bar(f"Kategori '{cat_to_delete}' ve içerdiği tüm notlar silindi.")
            else:
                delete_cat_dialog.open = False
                page.update()
                show_snack_bar("Failed to delete category.", color=ft.Colors.ERROR)
                
        delete_cat_dialog.title = ft.Text("Kategoriyi Sil")
        delete_cat_dialog.content = ft.Text("Bu kategoriyi ve içerdiği tüm notları silmek istediğinizden emin misiniz? Bu işlem geri alınamaz.")
        delete_cat_dialog.actions = [
            ft.TextButton("İptal", on_click=lambda e: close_dialog(delete_cat_dialog)),
            ft.TextButton("Sil", on_click=yes_click)
        ]
        delete_cat_dialog.open = True
        page.update()

    delete_category_click = delete_category_action

    def display_linked_notes():
        linked_notes_listview.controls.clear()
        if current_note_id:
            linked_note_ids = db_manager.get_note_links(current_note_id)
            if linked_note_ids:
                for linked_id in linked_note_ids:
                    note_data = db_manager.get_note(linked_id)
                    if note_data:
                        linked_title = note_data[1]
                        
                        def make_open_handler(nid=linked_id):
                            return lambda e: open_note_by_id(nid)
                            
                        def make_unlink_handler(nid=linked_id, title=linked_title):
                            return lambda e: unlink_note_confirm(nid, title)
                            
                        item = ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.Icons.LINK, color=ft.Colors.PRIMARY),
                                ft.Text(linked_title, color=ft.Colors.ON_SURFACE, weight=ft.FontWeight.W_500, expand=True),
                                ft.IconButton(
                                    icon=ft.Icons.LINK_OFF,
                                    icon_color=ft.Colors.ERROR,
                                    on_click=make_unlink_handler(),
                                    icon_size=18,
                                    tooltip="Unlink Note"
                                )
                            ]),
                            padding=5,
                            border_radius=5,
                            on_click=make_open_handler(),
                            on_hover=lambda e: setattr(e.control, 'bgcolor', ft.Colors.ON_INVERSE_SURFACE if e.data == "true" else ft.Colors.TRANSPARENT) or e.control.update()
                        )
                        linked_notes_listview.controls.append(item)
            else:
                linked_notes_listview.controls.append(ft.Text("No linked notes.", color=ft.Colors.ON_SURFACE_VARIANT, italic=True))
        else:
            linked_notes_listview.controls.append(ft.Text("Select a note to see its links.", color=ft.Colors.ON_SURFACE_VARIANT, italic=True))
        linked_notes_listview.update()

    def unlink_note_confirm(target_note_id, target_title):
        def yes_click(e):
            success = db_manager.delete_note_link(current_note_id, target_note_id)
            if success:
                unlink_dialog.open = False
                if hasattr(mind_map_widget, 'invalidate_cache'):
                    mind_map_widget.invalidate_cache()
                display_linked_notes()
                update_mind_map()
                page.update()
                show_snack_bar(f"Successfully unlinked '{target_title}'.")
            else:
                unlink_dialog.open = False
                page.update()
                show_snack_bar(f"Failed to unlink note: {target_title}.", color=ft.Colors.ERROR)
                
        unlink_dialog.content = ft.Text(f"Are you sure you want to unlink '{target_title}' from the current note?")
        unlink_dialog.actions = [
            ft.TextButton("Yes", on_click=yes_click),
            ft.TextButton("No", on_click=lambda e: close_dialog(unlink_dialog))
        ]
        unlink_dialog.open = True
        page.update()

    def link_note_action(e):
        if not current_note_id:
            show_snack_bar("Please select or save a source note first.", color=ft.Colors.ERROR)
            return
        link_search_field.value = ""
        load_linkable_notes()
        link_note_dialog.open = True
        page.update()

    def update_mind_map():
        log_debug("DEBUG: update_mind_map called.")
        all_notes_metadata, _ = note_manager.load_all_notes_metadata(db_manager)
        all_links = db_manager.get_all_note_links()
        
        selected_category = category_dropdown.value or ""
        
        filtered_notes_metadata = []
        filtered_note_ids = set()
        for note_id, title, category_path in all_notes_metadata:
            if selected_category == "" or category_path == selected_category:
                filtered_notes_metadata.append((note_id, title, category_path))
                filtered_note_ids.add(note_id)
                
        filtered_links = []
        for source_id, target_id in all_links:
            if source_id in filtered_note_ids and target_id in filtered_note_ids:
                filtered_links.append((source_id, target_id))
                
        mind_map_widget.update_map(filtered_notes_metadata, filtered_links, current_note_id)

    # ASYNCHRONOUS PDF NOTE GENERATION CALLBACKS
    # ------------------------------------------
    def handle_ai_generation_finished(generated_notes):
        # Callback safely triggered in UI context
        loading_dialog.open = False
        
        if generated_notes:
            nonlocal current_note_category
            current_note_category = ""
            if hasattr(mind_map_widget, 'invalidate_cache'):
                mind_map_widget.invalidate_cache()
            load_categories(select_category="")
            load_notes(category_to_select="")
            display_linked_notes()
            update_mind_map()
            show_snack_bar(f"{len(generated_notes)} notes were successfully generated and saved!")
        else:
            show_snack_bar("No notes were generated by the AI.", color=ft.Colors.TERTIARY)
        page.update()

    def handle_ai_generation_error(message):
        # Callback safely triggered in UI context
        loading_dialog.open = False
        
        error_dialog.title = ft.Text("AI Note Generation Error")
        error_dialog.content = ft.Text(f"An error occurred during AI note generation:\n{message}")
        error_dialog.actions = [
            ft.TextButton("OK", on_click=lambda e: close_dialog(error_dialog))
        ]
        error_dialog.open = True
        show_snack_bar(f"Error: {message}", color=ft.Colors.ERROR)
        page.update()

    def run_ai_generation_thread(extracted_text):
        try:
            worker = AiNoteGeneratorWorker(
                extracted_text,
                on_finished=handle_ai_generation_finished,
                on_error=handle_ai_generation_error
            )
            worker.run()
        except Exception as e:
            handle_ai_generation_error(str(e))

    pdf_file_picker = ft.FilePicker()
    if hasattr(page, 'services'):
        page.services.append(pdf_file_picker)
    else:
        page.overlay.append(pdf_file_picker)

    async def trigger_pdf_generation(e):
        files = await pdf_file_picker.pick_files(
            dialog_title="Select PDF File",
            allowed_extensions=["pdf"]
        )
        if files:
            pdf_path = files[0].path
            
            # Show progress bar dialog
            loading_dialog.title = ft.Text("Generating AI Notes From PDF")
            loading_dialog.content = ft.Row([
                ft.ProgressRing(),
                ft.Text("Extracting text from PDF... This may take a moment.")
            ], spacing=20, alignment=ft.MainAxisAlignment.CENTER)
            loading_dialog.open = True
            page.update()
            
            # Extract text
            try:
                extracted_text = pdf_processor.extract_text_from_pdf(pdf_path)
            except Exception as ex:
                loading_dialog.open = False
                page.update()
                show_snack_bar(f"Failed to read PDF: {ex}", color=ft.Colors.ERROR)
                return
            
            if extracted_text and extracted_text.strip():
                # Update text to reflect AI generation step
                loading_dialog.content = ft.Row([
                    ft.ProgressRing(),
                    ft.Text("Generating notes with AI... This may take longer.")
                ], spacing=20, alignment=ft.MainAxisAlignment.CENTER)
                loading_dialog.update()
                
                # Start AI note generation in background thread
                threading.Thread(
                    target=run_ai_generation_thread,
                    args=(extracted_text,),
                    daemon=True
                ).start()
            else:
                loading_dialog.open = False
                page.update()
                show_snack_bar("Selected PDF file is empty or contains no readable text.", color=ft.Colors.ERROR)
        else:
            show_snack_bar("No PDF file selected.", color=ft.Colors.TERTIARY)

    # Theme toggle handler
    def toggle_theme(e):
        if page.theme_mode == ft.ThemeMode.DARK:
            page.theme_mode = ft.ThemeMode.LIGHT
            db_manager.set_setting("UI_THEME", "Light")
            theme_btn.icon = ft.Icons.DARK_MODE
            theme_btn.tooltip = "Switch to Dark Mode"
        else:
            page.theme_mode = ft.ThemeMode.DARK
            db_manager.set_setting("UI_THEME", "Dark")
            theme_btn.icon = ft.Icons.LIGHT_MODE
            theme_btn.tooltip = "Switch to Light Mode"
        page.update()




    # Auto Save Setting
    auto_save_val = db_manager.get_setting("AUTO_SAVE")
    if auto_save_val is None:
        auto_save_val = "True"
        db_manager.set_setting("AUTO_SAVE", "True")
        
    def toggle_auto_save(e):
        val = "True" if auto_save_switch.value else "False"
        db_manager.set_setting("AUTO_SAVE", val)
        
    auto_save_switch = ft.Switch(label="Auto Save", value=(auto_save_val == "True"), on_change=toggle_auto_save)

    # THEME AND COMPONENT ICON INITIALIZATION
    theme_btn = ft.IconButton(
        icon=ft.Icons.LIGHT_MODE if page.theme_mode == ft.ThemeMode.DARK else ft.Icons.DARK_MODE,
        tooltip="Switch to Light Mode" if page.theme_mode == ft.ThemeMode.DARK else "Switch to Dark Mode",
        on_click=toggle_theme
    )

    # Settings Dialog
    def save_api_key_settings(e):
        api_key = api_key_field.value.strip()
        if api_key:
            dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
            set_key(dotenv_path, "GEMINI_API_KEY", api_key)
            show_snack_bar("Gemini API Key saved successfully.")
        
        settings_dialog.open = False
        page.update()

    settings_dialog = ft.AlertDialog(
        title=ft.Text("Settings"),
        content=ft.Column([
            ft.Row([ft.Text("Theme Mode", weight=ft.FontWeight.BOLD), theme_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Text("Auto Save", weight=ft.FontWeight.BOLD), auto_save_switch], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(),
            ft.Text("Gemini API Configuration", weight=ft.FontWeight.BOLD),
            ft.Text("Your API Key will be securely saved to your local .env file.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            api_key_field
        ], tight=True, spacing=15, width=400),
        actions=[
            ft.TextButton("Save", on_click=save_api_key_settings),
            ft.TextButton("Cancel", on_click=lambda e: close_dialog(settings_dialog))
        ],
        actions_alignment=ft.MainAxisAlignment.END
    )
    page.overlay.append(settings_dialog)

    def trigger_settings_dialog(e):
        settings_dialog.open = True
        page.update()

    # INTERACTION BUTTONS ROW (MIDDLE CONTAINER UNDER SPLITSCREEN)
    interaction_buttons = ft.Row([
        ft.Button(
            content="New Note",
            icon=ft.Icons.ADD,
            icon_color=ft.Colors.PRIMARY,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            color=ft.Colors.PRIMARY,
            on_click=lambda e: new_note(),
            expand=True
        ),
        ft.Button(
            content="Save Note",
            icon=ft.Icons.SAVE,
            icon_color=ft.Colors.SECONDARY,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            color=ft.Colors.SECONDARY,
            on_click=save_note_action,
            expand=True
        ),
        ft.Button(
            content="Delete Note",
            icon=ft.Icons.DELETE,
            icon_color=ft.Colors.ERROR,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            color=ft.Colors.ERROR,
            on_click=delete_current_note,
            expand=True
        ),
        ft.Button(
            content="Link Note",
            icon=ft.Icons.LINK,
            icon_color=ft.Colors.TERTIARY,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            color=ft.Colors.TERTIARY,
            on_click=link_note_action,
            expand=True
        ),
        ft.Button(
            content="Generate AI Notes",
            icon=ft.Icons.AUTO_AWESOME,
            icon_color=ft.Colors.TERTIARY,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            color=ft.Colors.TERTIARY,
            on_click=trigger_pdf_generation,
            expand=True
        )
    ], spacing=10)

    # Panel Sürükleme Callback'leri (Dinamik Boyutlandırma)
    def left_sidebar_drag(e: ft.DragUpdateEvent):
        delta = e.local_delta.x if e.local_delta else 0
        new_width = left_sidebar.width + delta
        if 180 <= new_width <= 600:
            left_sidebar.width = new_width
            page.update()

    def right_sidebar_drag(e: ft.DragUpdateEvent):
        delta = e.local_delta.x if e.local_delta else 0
        new_width = right_sidebar.width - delta
        if 200 <= new_width <= 700:
            right_sidebar.width = new_width
            page.update()

    def mind_map_drag(e: ft.DragUpdateEvent):
        delta = e.local_delta.y if e.local_delta else 0
        new_height = mind_map_container.height + delta
        if 150 <= new_height <= 600:
            mind_map_container.height = new_height
            page.update()

    # Splitter Hover Olay Dinleyicileri
    def vertical_splitter_hover(e):
        if e.data == "true":
            e.control.content.width = 3
            e.control.content.bgcolor = ft.Colors.PRIMARY
        else:
            e.control.content.width = 2
            e.control.content.bgcolor = ft.Colors.OUTLINE_VARIANT
        e.control.update()

    def horizontal_splitter_hover(e):
        if e.data == "true":
            e.control.content.height = 3
            e.control.content.bgcolor = ft.Colors.PRIMARY
        else:
            e.control.content.height = 2
            e.control.content.bgcolor = ft.Colors.OUTLINE_VARIANT
        e.control.update()

    # Splitter (Ayırıcı) Bileşenleri
    left_splitter = ft.GestureDetector(
        content=ft.Container(
            width=6,
            bgcolor="transparent",
            alignment=ft.Alignment(0, 0),
            content=ft.Container(
                width=2,
                bgcolor=ft.Colors.OUTLINE_VARIANT,
                border_radius=1,
            ),
            on_hover=vertical_splitter_hover
        ),
        mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT,
        on_pan_update=left_sidebar_drag
    )

    right_splitter = ft.GestureDetector(
        content=ft.Container(
            width=6,
            bgcolor="transparent",
            alignment=ft.Alignment(0, 0),
            content=ft.Container(
                width=2,
                bgcolor=ft.Colors.OUTLINE_VARIANT,
                border_radius=1,
            ),
            on_hover=vertical_splitter_hover
        ),
        mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT,
        on_pan_update=right_sidebar_drag
    )

    mind_map_splitter = ft.GestureDetector(
        content=ft.Container(
            height=6,
            bgcolor="transparent",
            alignment=ft.Alignment(0, 0),
            content=ft.Container(
                height=2,
                bgcolor=ft.Colors.OUTLINE_VARIANT,
                border_radius=1,
            ),
            on_hover=horizontal_splitter_hover
        ),
        mouse_cursor=ft.MouseCursor.RESIZE_UP_DOWN,
        on_pan_update=mind_map_drag
    )

    # ASSEMBLE PAGE PANELS
    # ---------------------
    # Left Sidebar Panel
    left_sidebar = ft.Container(
        content=ft.Column([
            ft.Text("Zettelkasten AI Notes", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
            ft.Divider(),
            ft.Row([
                category_dropdown,
                ft.IconButton(ft.Icons.ADD_BOX, on_click=lambda e: (setattr(new_category_field, 'value', '') or setattr(new_category_dialog, 'open', True) or page.update()), tooltip="New Category"),
                ft.IconButton(ft.Icons.DELETE_FOREVER, on_click=delete_category_action, tooltip="Delete Category")
            ], spacing=5),
            search_textfield,
            ft.Divider(),
            notes_listview,
            ft.Divider(),
            ft.Row([
                note_count_label,
                ft.Row([
                    ft.IconButton(ft.Icons.SETTINGS, on_click=trigger_settings_dialog, tooltip="Settings")
                ])
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        ], expand=True),
        width=300,
        padding=15,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=12
    )

    # Middle Editor/Preview Panel
    middle_editor = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        ft.Text("Editor", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                        ft.Divider(height=1),
                        editor_textfield
                    ], expand=True),
                    expand=True,
                    padding=10,
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=8,
                    border=ft.Border.all(1, ft.Colors.ON_INVERSE_SURFACE)
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("Live Preview", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.SECONDARY),
                        ft.Divider(height=1),
                        ft.Container(
                            content=ft.Column([preview_markdown], scroll=ft.ScrollMode.AUTO),
                            expand=True
                        )
                    ], expand=True),
                    expand=True,
                    padding=10,
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=8,
                    border=ft.Border.all(1, ft.Colors.ON_INVERSE_SURFACE)
                )
            ], expand=True, spacing=10),
            interaction_buttons
        ], expand=True, spacing=15),
        expand=True,
        padding=15,
        bgcolor="#0a0e17",
        border_radius=12
    )

    # Mind Map Container
    mind_map_container = ft.Container(
        content=mind_map_widget,
        height=350,
        border_radius=8,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        border=ft.Border.all(1.5, ft.Colors.ON_INVERSE_SURFACE)
    )

    # Right Sidebar (Mind Map and Links)
    right_sidebar = ft.Container(
        content=ft.Column([
            ft.Text("Interactive Mind Map", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
            ft.Divider(),
            mind_map_container,
            mind_map_splitter,
            ft.Divider(),
            ft.Text("Linked Connections", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
            linked_notes_listview
        ], expand=True),
        width=350,
        padding=15,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=12
    )

    # Final Layout Assembly in Main Window
    page.add(
        ft.Row([
            left_sidebar,
            left_splitter,
            middle_editor,
            right_splitter,
            right_sidebar
        ], expand=True, spacing=5)
    )
    
    # Initial loads
    load_categories()
    load_notes()
    display_linked_notes()

if __name__ == '__main__':
    ft.run(main)