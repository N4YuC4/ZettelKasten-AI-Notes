# main.py
#
# Bu dosya, Zettelkasten AI Notes uygulamasının ana giriş noktasıdır.
# PyQt5 kullanarak bir masaüstü uygulaması arayüzü sağlar ve not yönetimi,
# Markdown önizlemesi, kategori yönetimi, PDF'ten not oluşturma ve
# notlar arası bağlantı kurma gibi temel işlevleri içerir.

import sys # Sistemle ilgili işlevler için (örn. uygulama çıkışı)
import os # Dosya sistemi işlemleri için (örn. dosya yolu birleştirme)
from dotenv import set_key # .env dosyasındaki ortam değişkenlerini ayarlamak için
from logger import *
import markdown # Markdown metnini HTML'e dönüştürmek için

# PyQt5 GUI bileşenleri için gerekli modüller
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QMainWindow,
    QAction, QListWidget, QSplitter, QMessageBox, QInputDialog, QHBoxLayout,
    QComboBox, QStyle, QMenu, QProgressDialog, QDialog, QLabel, QLineEdit,
    QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread # Qt temel tipleri ve çoklu iş parçacığı (threading) için

# Uygulamanın diğer modülleri
import database_manager # Veritabanı işlemleri için
import note_manager # Notların kaydedilmesi, yüklenmesi, yeniden adlandırılması ve silinmesi gibi not yönetimi işlevleri için
import pdf_processor # PDF dosyalarından metin çıkarmak için
from ai_note_generator_worker import AiNoteGeneratorWorker # Yapay zeka ile not oluşturma işlemini ayrı bir iş parçacığında yürütmek için

# Dosya seçimi için Tkinter kütüphanesi (PyQt5 ile entegre değil, ayrı bir pencere açar)
import tkinter as tk
from tkinter import filedialog

# NoteSelectionDialog sınıfı, kullanıcının mevcut notlar arasından bir not seçmesini sağlayan bir iletişim kutusu.
# Bu, özellikle notları birbirine bağlarken kullanılır.
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QLabel, QLineEdit, QHBoxLayout, QListWidgetItem


# NoteSelectionDialog sınıfı, kullanıcının mevcut notlar arasından bir not seçmesini sağlayan bir iletişim kutusu.
# Bu, özellikle notları birbirine bağlarken kullanılır.
class NoteSelectionDialog(QDialog):
    # __init__ metodu, iletişim kutusunu başlatır.
    # db_manager: Veritabanı işlemleri için DatabaseManager nesnesi.
    # current_note_id: Bağlantı kurulacak mevcut notun ID'si (kendisiyle bağlantı kurmayı engellemek için).
    # parent: Bu iletişim kutusunun ebeveyn widget'ı.
    def __init__(self, db_manager, current_note_id, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Note to Link") # Pencere başlığını ayarla
        self.setGeometry(200, 200, 400, 600) # Pencerenin konumunu ve boyutunu ayarla

        self.db_manager = db_manager # Veritabanı yöneticisini sakla
        self.current_note_id = current_note_id # Mevcut notun ID'sini sakla
        self.selected_note_id = None # Seçilen notun ID'si
        self.selected_note_title = None # Seçilen notun başlığı

        self.init_ui() # Kullanıcı arayüzünü başlat
        self.load_notes() # Notları yükle

    # init_ui metodu, iletişim kutusunun kullanıcı arayüzünü oluşturur.
    def init_ui(self):
        self.main_layout = QVBoxLayout(self) # Ana dikey düzenleyici

        # Arama kutusu
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search notes...") # Yer tutucu metin
        self.search_input.textChanged.connect(self.load_notes) # Metin değiştiğinde notları yeniden yükle
        self.main_layout.addWidget(self.search_input) # Düzenleyiciye ekle

        # Not listesi widget'ı
        self.notes_list_widget = QListWidget()
        self.notes_list_widget.itemClicked.connect(self.on_note_selected) # Bir öğeye tıklandığında olayı bağla
        self.main_layout.addWidget(self.notes_list_widget) # Düzenleyiciye ekle

        # Butonlar için yatay düzenleyici
        self.button_layout = QHBoxLayout()
        self.select_button = QPushButton("Select") # Seç butonu
        self.select_button.clicked.connect(self.accept) # Tıklandığında iletişim kutusunu kabul et
        self.select_button.setEnabled(False) # Başlangıçta devre dışı bırak
        self.button_layout.addWidget(self.select_button) # Düzenleyiciye ekle

        self.cancel_button = QPushButton("Cancel") # İptal butonu
        self.cancel_button.clicked.connect(self.reject) # Tıklandığında iletişim kutusunu reddet
        self.button_layout.addWidget(self.cancel_button) # Düzenleyiciye ekle

        self.main_layout.addLayout(self.button_layout) # Ana düzenleyiciye buton düzenleyiciyi ekle

    # load_notes metodu, veritabanından notları yükler ve listeler.
    # Arama kutusundaki metne göre filtreleme yapar.
    def load_notes(self):
        self.notes_list_widget.clear() # Mevcut listeyi temizle
        search_text = self.search_input.text().lower() # Arama metnini al ve küçük harfe dönüştür

        # Tüm notların meta verilerini yükle
        all_notes_metadata, _ = note_manager.load_all_notes_metadata(self.db_manager)
        
        # Her not için
        for note_id, title, _ in all_notes_metadata:
            if note_id == self.current_note_id: # Mevcut notu listede gösterme (kendisiyle bağlantı kurmayı engelle)
                continue
            if search_text in title.lower(): # Arama metni başlıkta varsa
                item = QListWidgetItem(title) # Yeni bir liste öğesi oluştur
                item.setData(Qt.UserRole, note_id) # Not ID'sini öğenin verisine sakla
                self.notes_list_widget.addItem(item) # Listeye ekle

    # on_note_selected metodu, listeden bir not seçildiğinde çağrılır.
    # item: Seçilen QListWidgetItem nesnesi.
    def on_note_selected(self, item):
        self.selected_note_id = item.data(Qt.UserRole) # Seçilen notun ID'sini al
        self.selected_note_title = item.text() # Seçilen notun başlığını al
        self.select_button.setEnabled(True) # Seç butonunu etkinleştir


class ZettelkastenApp(QMainWindow):
    # __init__ metodu, ana uygulama penceresini başlatır.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zettelkasten AI Notes") # Pencere başlığını ayarla
        self.setGeometry(100, 100, 1200, 800) # Pencerenin konumunu ve boyutunu ayarla

        # Veritabanı yöneticisi nesnesini oluştur
        self.db_manager = database_manager.DatabaseManager()
        self.current_note_id = None # Şu anda açık olan notun ID'si
        self.current_note_category = "" # Şu anda açık olan notun kategorisi
        # Görüntülenen başlıkları not ID'lerine eşleyen sözlük (not listesi için)
        self.displayed_title_to_note_id = {}
        # Not ID'lerini kategorilere eşleyen sözlük
        self.note_id_to_category = {}

        self.init_ui() # Kullanıcı arayüzünü başlat
        self.load_notes() # Notları yükle

        # Load and apply theme from settings
        saved_theme = self.db_manager.get_setting("UI_THEME")
        if saved_theme:
            self.apply_theme(saved_theme)
        else:
            self.apply_theme("Light") # Default theme

    def apply_theme(self, theme_name):
        stylesheet_path = os.path.join(os.path.dirname(__file__), f'{theme_name.lower()}_theme.qss')
        if os.path.exists(stylesheet_path):
            with open(stylesheet_path, "r") as f:
                QApplication.instance().setStyleSheet(f.read())
            self.db_manager.set_setting("UI_THEME", theme_name)
        else:
            QMessageBox.warning(self, "Theme Error", f"Theme file not found: {stylesheet_path}")

    # init_ui metodu, ana pencerenin kullanıcı arayüzünü oluşturur.
    def init_ui(self):
        self.central_widget = QWidget() # Ana pencerenin merkezi widget'ı
        self.setCentralWidget(self.central_widget) # Merkezi widget'ı ayarla
        self.main_layout = QVBoxLayout(self.central_widget) # Ana dikey düzenleyici

        # Butonlar için yatay düzenleyici
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout) # Ana düzenleyiciye ekle

        self.note_count_label = QLabel(f"{self.current_note_category} not sayısı: {self.db_manager.note_count(self.current_note_category)}") # Kategori etiketi
        self.button_layout.addWidget(self.note_count_label) # Buton düzenleyiciye ekle

        # Kategori seçim kutusu (ComboBox)
        self.category_combo_box = QComboBox()
        self.category_combo_box.addItem("All Notes") # Varsayılan olarak "Tüm Notlar" seçeneğini ekle
        # Seçim değiştiğinde load_notes metodunu çağır
        self.category_combo_box.currentIndexChanged.connect(self.load_notes) 
        self.button_layout.addWidget(self.category_combo_box) # Buton düzenleyiciye ekle

        style = QApplication.instance().style() # Uygulamanın stilini al (ikonlar için)

        # Yeni Kategori Butonu
        self.new_category_button = QPushButton(style.standardIcon(QStyle.SP_DirOpenIcon), "New Category") 
        self.new_category_button.clicked.connect(self.create_new_category) # Tıklandığında yeni kategori oluştur
        self.button_layout.addWidget(self.new_category_button) # Buton düzenleyiciye ekle

        # Kategori Sil Butonu
        self.delete_category_button = QPushButton(style.standardIcon(QStyle.SP_TrashIcon), "Delete Category")
        self.delete_category_button.clicked.connect(self.delete_category) # Tıklandığında kategoriyi sil
        self.button_layout.addWidget(self.delete_category_button) # Buton düzenleyiciye ekle

        # Yeni Not Butonu
        self.new_button = QPushButton(style.standardIcon(QStyle.SP_FileIcon), "New Note") 
        self.new_button.clicked.connect(self.new_note) # Tıklandığında yeni not oluştur
        self.button_layout.addWidget(self.new_button) # Buton düzenleyiciye ekle

        # Notu Kaydet Butonu
        self.save_button = QPushButton(style.standardIcon(QStyle.SP_DialogSaveButton), "Save Note")
        self.save_button.clicked.connect(self.save_note) # Tıklandığında notu kaydet
        self.button_layout.addWidget(self.save_button) # Buton düzenleyiciye ekle

        # Notu Yeniden Adlandır Butonu
        self.rename_button = QPushButton(style.standardIcon(QStyle.SP_DialogResetButton), "Rename Note") 
        self.rename_button.clicked.connect(self.rename_note) # Tıklandığında notu yeniden adlandır
        self.button_layout.addWidget(self.rename_button) # Buton düzenleyiciye ekle

        # Notu Sil Butonu
        self.delete_button = QPushButton(style.standardIcon(QStyle.SP_TrashIcon), "Delete Note")
        self.delete_button.clicked.connect(self.delete_note) # Tıklandığında notu sil
        self.button_layout.addWidget(self.delete_button) # Buton düzenleyiciye ekle

        # PDF'ten AI Notları Oluştur Butonu
        self.generate_notes_button = QPushButton(style.standardIcon(QStyle.SP_FileDialogDetailedView), "Generate AI Notes from PDF")
        self.generate_notes_button.clicked.connect(self.generate_notes_from_pdf) # Tıklandığında PDF'ten not oluştur
        self.button_layout.addWidget(self.generate_notes_button) # Buton düzenleyiciye ekle

        # Notu Bağla Butonu
        self.link_note_button = QPushButton(style.standardIcon(QStyle.SP_DialogYesButton), "Link Note") 
        self.link_note_button.clicked.connect(self.link_note_action) # Tıklandığında notu bağla
        self.button_layout.addWidget(self.link_note_button) # Buton düzenleyiciye ekle

        # Ana ayırıcı (splitter) - Not listesi ve düzenleyici/önizleme arasında
        self.splitter = QSplitter(Qt.Horizontal) # Yatay ayırıcı
        self.main_layout.addWidget(self.splitter) # Ana düzenleyiciye ekle

        # Not listesi widget'ı
        self.notes_list_widget = QListWidget()
        self.notes_list_widget.itemClicked.connect(self.open_selected_note) # Bir öğeye tıklandığında notu aç
        self.notes_list_widget.setContextMenuPolicy(Qt.CustomContextMenu) # Özel bağlam menüsü politikasını ayarla
        # Bağlam menüsü istendiğinde _show_note_context_menu metodunu çağır
        self.notes_list_widget.customContextMenuRequested.connect(self._show_note_context_menu)
        self.splitter.addWidget(self.notes_list_widget) # Ayırıcıya ekle

        # Düzenleyici ve önizleme arasında dikey ayırıcı
        self.editor_preview_splitter = QSplitter(Qt.Vertical) # Dikey ayırıcı
        self.splitter.addWidget(self.editor_preview_splitter) # Ana ayırıcıya ekle

        # Not düzenleyici (metin alanı)
        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Notlarınızı buraya yazın...") # Yer tutucu metin
        self.editor.textChanged.connect(self.update_preview) # Metin değiştiğinde önizlemeyi güncelle
        self.editor_preview_splitter.addWidget(self.editor) # Ayırıcıya ekle

        # Markdown önizleme alanı
        self.preview = QTextEdit()
        self.preview.setReadOnly(True) # Sadece okunabilir yap
        self.preview.setPlaceholderText("Markdown önizlemesi burada görünecek...") # Yer tutucu metin
        self.editor_preview_splitter.addWidget(self.preview) # Ayırıcıya ekle

        # Bağlantılı Notlar Listesi
        self.linked_notes_list_widget = QListWidget()
        self.linked_notes_list_widget.itemClicked.connect(self.open_selected_note_from_link) # Bağlantılı nota tıklandığında aç
        self.linked_notes_list_widget.setContextMenuPolicy(Qt.CustomContextMenu) # Özel bağlam menüsü politikasını ayarla
        # Bağlam menüsü istendiğinde _show_linked_note_context_menu metodunu çağır
        self.linked_notes_list_widget.customContextMenuRequested.connect(self._show_linked_note_context_menu)
        self.editor_preview_splitter.addWidget(self.linked_notes_list_widget) # Ayırıcıya ekle

        self.display_linked_notes() # Bağlantılı notları göster

        self.create_menu_bar() # Menü çubuğunu oluştur

    # create_menu_bar metodu, uygulamanın menü çubuğunu oluşturur.
    def create_menu_bar(self):
        menubar = self.menuBar() # Menü çubuğunu al
        settings_menu = menubar.addMenu("Settings") # "Ayarlar" menüsünü ekle

        # Tema menüsü
        theme_menu = menubar.addMenu("Theme")
        light_theme_action = QAction("Light Theme", self)
        light_theme_action.triggered.connect(lambda: self.apply_theme("Light"))
        theme_menu.addAction(light_theme_action)

        dark_theme_action = QAction("Dark Theme", self)
        dark_theme_action.triggered.connect(lambda: self.apply_theme("Dark"))
        theme_menu.addAction(dark_theme_action)

        # Gemini API Anahtarı giriş eylemi
        gemini_api_key_action = QAction("Enter Gemini API Key", self)
        gemini_api_key_action.triggered.connect(self._show_api_key_dialog) # Tıklandığında API anahtarı iletişim kutusunu göster
        settings_menu.addAction(gemini_api_key_action) # Menüye eylemi ekle

    # _show_note_context_menu metodu, not listesindeki öğelere sağ tıklandığında açılan bağlam menüsünü gösterir.
    # position: Sağ tıklamanın gerçekleştiği konum.
    def _show_note_context_menu(self, position):
        item = self.notes_list_widget.itemAt(position) # Tıklanan konumdaki öğeyi al

        if item:
            self.notes_list_widget.setCurrentItem(item) # Tıklanan öğeyi seçili hale getir

            context_menu = QMenu(self) # Yeni bir bağlam menüsü oluştur

            rename_action = context_menu.addAction("Rename Note") # "Notu Yeniden Adlandır" eylemini ekle
            delete_action = context_menu.addAction("Delete Note") # "Notu Sil" eylemini ekle
            link_action = context_menu.addAction("Link to...") # "Bağla..." eylemini ekle

            # Menüyü göster ve seçilen eylemi al
            action = context_menu.exec_(self.notes_list_widget.mapToGlobal(position))

            # Seçilen eyleme göre ilgili metodu çağır
            if action == rename_action:
                self.rename_note()
            elif action == delete_action:
                self.delete_note()
            elif action == link_action:
                self.link_note_action()

    # _show_api_key_dialog metodu, Gemini API anahtarını girmek için bir iletişim kutusu gösterir
    # ve anahtarı .env dosyasına kaydeder.
    def _show_api_key_dialog(self):
        # Kullanıcıdan API anahtarını girmesini iste
        api_key, ok = QInputDialog.getText(self, "Gemini API Key", "Enter your Gemini API Key:")
        if ok and api_key: # Kullanıcı Tamam'a basar ve anahtar boş değilse
            # .env dosyasının yolunu belirle (main.py'nin bir üst dizininde)
            dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
            set_key(dotenv_path, "GEMINI_API_KEY", api_key) # Anahtarı .env dosyasına kaydet
            QMessageBox.information(self, "Gemini API Key", "Gemini API Key saved to .env file. Please restart the application for changes to take effect.")
        elif ok and not api_key: # Kullanıcı Tamam'a basar ama anahtar boşsa
            QMessageBox.warning(self, "Gemini API Key", "API Key cannot be empty.")

    # link_note_action metodu, seçili notu başka bir nota bağlamak için kullanılır.
    def link_note_action(self):
        if not self.current_note_id: # Mevcut bir not seçili değilse uyarı ver
            QMessageBox.warning(self, "Link Note", "Please select a source note first.")
            return

        # Not seçme iletişim kutusunu oluştur ve göster
        dialog = NoteSelectionDialog(self.db_manager, self.current_note_id, self)
        if dialog.exec_() == QDialog.Accepted: # Kullanıcı bir not seçip Tamam'a basarsa
            target_note_id = dialog.selected_note_id # Hedef notun ID'sini al
            target_note_title = dialog.selected_note_title # Hedef notun başlığını al

            # Hedef not ID'si varsa ve mevcut nottan farklıysa
            if target_note_id and target_note_id != self.current_note_id:
                # Not bağlantısını veritabanına ekle
                success = self.db_manager.insert_note_link(self.current_note_id, target_note_id)
                if success:
                    QMessageBox.information(self, "Link Note", f"Successfully linked to '{target_note_title}'.")
                    self.display_linked_notes() # Bağlantılı notlar listesini yenile
                else:
                    QMessageBox.warning(self, "Link Note", f"Link to '{target_note_title}' already exists or failed to create.")
            elif target_note_id == self.current_note_id: # Notu kendisine bağlamaya çalışırsa
                QMessageBox.warning(self, "Link Note", "Cannot link a note to itself.")

    # create_new_category metodu, yeni bir not kategorisi oluşturur.
    def create_new_category(self):
        # Kullanıcıdan kategori adını girmesini iste
        category_name, ok = QInputDialog.getText(self, "New Category", "Enter category name:")
        if ok and category_name: # Kullanıcı Tamam'a basar ve kategori adı boş değilse
            # Mevcut not meta verilerini ve kategorileri yükle
            all_notes_metadata, all_categories = note_manager.load_all_notes_metadata(self.db_manager)
            if category_name in all_categories: # Kategori zaten varsa uyarı ver
                QMessageBox.warning(self, "New Category", f"Category '{category_name}' already exists.")
                return

            self.new_note() # Yeni bir boş not oluştur
            self.current_note_category = category_name # Mevcut notun kategorisini ayarla
            # Editöre varsayılan metni ekle
            self.editor.setPlainText(f"# New note in {category_name}\n\nStart writing your note here...")
            self.save_note() # Notu kaydet
            QMessageBox.information(self, "New Category", f"Category '{category_name}' created and a new note added.")
            self.load_notes(category_to_select=category_name) # Notları yeniden yükle ve yeni kategoriyi seç

    # new_note metodu, düzenleyiciyi ve önizlemeyi temizleyerek yeni bir not oluşturmaya hazırlar.
    def new_note(self):
        self.editor.clear() # Editörü temizle
        self.preview.clear() # Önizlemeyi temizle
        self.current_note_id = None # Mevcut not ID'sini sıfırla
        # Mevcut kategori seçim kutusundan al, "All Notes" ise boş bırak
        self.current_note_category = self.category_combo_box.currentText() if self.category_combo_box.currentText() != "All Notes" else ""
        self.setWindowTitle("Zettelkasten AI Notes - New Note") # Pencere başlığını güncelle

    # save_note metodu, mevcut notun içeriğini veritabanına kaydeder veya günceller.
    def save_note(self):
        note_content = self.editor.toPlainText() # Editördeki metni al
        if not note_content.strip(): # Not içeriği boşsa uyarı ver
            QMessageBox.warning(self, "Save Note", "Cannot save empty note.")
            return

        # Kaydedilecek kategoriyi belirle
        category_to_save = self.current_note_category if self.current_note_category != "All Notes" else ""

        # Notu kaydet veya güncelle
        note_id, display_title = note_manager.save_note(
            self.db_manager, 
            self.current_note_id, 
            note_content, 
            category_to_save
        )

        if note_id and display_title: # Not başarıyla kaydedildiyse
            self.current_note_id = note_id # Mevcut not ID'sini güncelle
            self.current_note_category = category_to_save # Mevcut not kategorisini güncelle
            self.setWindowTitle(f"Zettelkasten AI Notes - {display_title}") # Pencere başlığını güncelle
            self.load_notes(category_to_select=category_to_save) # Notları yeniden yükle ve kategoriyi seç
        else:
            QMessageBox.critical(self, "Error", "Failed to save note.") # Kaydetme başarısız olursa hata mesajı göster

    # rename_note metodu, seçili notu yeniden adlandırır.
    def rename_note(self):
        selected_items = self.notes_list_widget.selectedItems() # Seçili öğeleri al
        if not selected_items: # Hiçbir öğe seçili değilse uyarı ver
            QMessageBox.information(self, "Rename Note", "Please select a note to rename.")
            return

        selected_display_title = selected_items[0].text() # Seçili notun görünen başlığını al
        # Görünen başlıktan not ID'sini al
        note_id_to_rename = self.displayed_title_to_note_id.get(selected_display_title)

        if not note_id_to_rename: # Not ID'si bulunamazsa hata mesajı göster
            QMessageBox.critical(self, "Error", f"Could not find note ID for note: {selected_display_title}")
            return

        # Kullanıcıdan yeni başlığı girmesini iste (mevcut başlık önceden doldurulmuş olarak)
        new_title, ok = QInputDialog.getText(self, "Rename Note", "Enter new title:", 
                                            text=selected_display_title)

        if ok and new_title: # Kullanıcı Tamam'a basar ve yeni başlık boş değilse
            # Notu yeniden adlandır
            success, new_display_title = note_manager.rename_note(
                self.db_manager, 
                note_id_to_rename, 
                new_title, 
                self.note_id_to_category.get(note_id_to_rename, "")) # Notun kategorisini al

            if success: # Yeniden adlandırma başarılıysa
                if self.current_note_id == note_id_to_rename: # Eğer yeniden adlandırılan not açık olan notsa
                    self.setWindowTitle(f"Zettelkasten AI Notes - {new_display_title}") # Pencere başlığını güncelle
                # Notları yeniden yükle ve ilgili kategoriyi seç
                self.load_notes(category_to_select=self.note_id_to_category.get(note_id_to_rename, "")) 
            else:
                QMessageBox.critical(self, "Error", f"Failed to rename note: {new_display_title}") # Hata mesajı göster
        elif ok and not new_title: # Kullanıcı Tamam'a basar ama yeni başlık boşsa
            QMessageBox.warning(self, "Rename Note", "New title cannot be empty.")

    # delete_note metodu, seçili notu veritabanından siler.
    def delete_note(self):
        selected_items = self.notes_list_widget.selectedItems() # Seçili öğeleri al
        if not selected_items: # Hiçbir öğe seçili değilse uyarı ver
            QMessageBox.information(self, "Delete Note", "Please select a note to delete.")
            return

        selected_display_title = selected_items[0].text() # Seçili notun görünen başlığını al
        # Görünen başlıktan not ID'sini al
        note_id_to_delete = self.displayed_title_to_note_id.get(selected_display_title)

        if not note_id_to_delete: # Not ID'si bulunamazsa hata mesajı göster
            QMessageBox.critical(self, "Error", f"Could not find note ID for note: {selected_display_title}")
            return

        # Kullanıcıya silme işlemini onaylaması için soru sor
        reply = QMessageBox.question(self, 'Delete Note', 
                                     f"Are you sure you want to delete '{selected_display_title}'?\nThis action cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes: # Kullanıcı onaylarsa
            success = note_manager.delete_note(self.db_manager, note_id_to_delete) # Notu sil
            if success:
                self.new_note() # Yeni bir boş not oluştur (düzenleyiciyi temizle)
                # Notları yeniden yükle (mevcut kategori seçili olarak)
                self.load_notes(category_to_select=self.category_combo_box.currentText() if self.category_combo_box.currentText() != "All Notes" else "") 
                self.display_linked_notes() # Bağlantılı notlar listesini yenile
            else:
                QMessageBox.critical(self, "Error", "Failed to delete note.") # Hata mesajı göster

    # delete_category metodu, seçili kategoriyi ve bu kategoriye ait tüm notları siler.
    def delete_category(self):
        selected_category = self.category_combo_box.currentText() # Seçili kategoriyi al
        if not selected_category: # Kategori seçili değilse uyarı ver
            QMessageBox.information(self, "Delete Category", "Please select a category to delete.")
            return

        # Bu satır gereksiz gibi görünüyor, muhtemelen bir hata veya eksik kod.
        selected_category

        if not selected_category : # Kategori adı boşsa hata mesajı göster (önceki kontrolle çakışıyor)
            QMessageBox.critical(self, "Error", f"Could not find category for note: {selected_category}")
            return
        elif selected_category == "All Notes": # "Tüm Notlar" kategorisi silinemez
            QMessageBox.information(self, "Delete Category", "Cannot delete 'All Notes' category.")
            return

        # Kullanıcıya silme işlemini onaylaması için soru sor
        reply = QMessageBox.question(self, 'Delete Category',
                                     f"Are you sure you want to delete category '{selected_category}'?\nThis action cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes: # Kullanıcı onaylarsa
            success = self.db_manager.delete_category(selected_category) # Kategoriyi sil
            if success:
                self.new_note() # Yeni bir boş not oluştur (düzenleyiciyi temizle)
                # Notları yeniden yükle (mevcut kategori seçili olarak)
                self.load_notes(category_to_select=self.category_combo_box.currentText() if self.category_combo_box.currentText() != "All Notes" else "")  
            else:
                QMessageBox.critical(self, "Error", "Failed to delete category.") # Hata mesajı göster

    # generate_notes_from_pdf metodu, bir PDF dosyasından metin çıkarır ve bu metni kullanarak
    # yapay zeka ile notlar oluşturur. Bu işlem ayrı bir iş parçacığında (thread) yapılır.
    def generate_notes_from_pdf(self):
        self.hide() # Ana pencereyi gizle (Tkinter dosya iletişim kutusu için)
        root = tk.Tk() # Tkinter kök penceresi oluştur
        root.withdraw() # Kök pencereyi gizle

        # Kullanıcıdan PDF dosyası seçmesini iste
        pdf_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf")]
        )
        root.destroy() # Tkinter kök penceresini yok et
        self.show() # Ana pencereyi tekrar göster

        if pdf_path: # Bir PDF dosyası seçildiyse
            # İlerleme iletişim kutusunu göster (PDF metin çıkarma için)
            self.loading_dialog = QProgressDialog("Extracting text from PDF... This may take a moment.", None, 0, 0, self)
            self.loading_dialog.setWindowModality(Qt.WindowModal) # Modsal pencere yap
            self.loading_dialog.setCancelButton(None) # İptal butonunu kaldır
            self.loading_dialog.setWindowTitle("Generating AI Notes From PDF") # Başlığı ayarla
            self.loading_dialog.show() # İletişim kutusunu göster
            QApplication.processEvents() # GUI olaylarını işle

            extracted_text = pdf_processor.extract_text_from_pdf(pdf_path) # PDF'ten metin çıkar

            if extracted_text: # Metin başarıyla çıkarıldıysa
                # İlerleme iletişim kutusunun metnini güncelle (AI not oluşturma için)
                self.loading_dialog.setLabelText("Generating notes with AI... This may take longer.")
                QApplication.processEvents() # GUI olaylarını işle

                self.thread = QThread() # Yeni bir iş parçacığı oluştur
                self.worker = AiNoteGeneratorWorker(extracted_text) # AI not oluşturucu işçisi oluştur
                self.worker.moveToThread(self.thread) # İşçiyi iş parçacığına taşı

                # İş parçacığı sinyallerini ve işçi slotlarını bağla
                self.thread.started.connect(self.worker.run) # İş parçacığı başladığında işçinin run metodunu çağır
                self.worker.finished.connect(self.handle_ai_generation_finished) # İşçi bittiğinde handle_ai_generation_finished'i çağır
                self.worker.error.connect(self.handle_ai_generation_error) # İşçi hata verdiğinde handle_ai_generation_error'ı çağır
                self.worker.finished.connect(self.thread.quit) # İşçi bittiğinde iş parçacığını sonlandır
                self.worker.error.connect(self.thread.quit) # İşçi hata verdiğinde iş parçacığını sonlandır
                self.worker.finished.connect(self.worker.deleteLater) # İşçi bittiğinde işçiyi sil
                self.worker.error.connect(self.worker.deleteLater) # İşçi hata verdiğinde işçiyi sil
                self.thread.finished.connect(self.thread.deleteLater) # İş parçacığı bittiğinde iş parçacığını sil

                self.thread.start() # İş parçacığını başlat
            else:
                QMessageBox.warning(self, "PDF Processing Failed", "Could not extract text from the selected PDF file.")
        else:
            QMessageBox.information(self, "PDF Selection Cancelled", "No PDF file selected.")

    # load_notes metodu, veritabanından notları ve kategorileri yükler ve not listesini günceller.
    # index: Seçilecek kategori ComboBox indeksini belirtir.
    # category_to_select: Seçilecek kategori adını belirtir.
    def load_notes(self, index=None, category_to_select=None): 
        print(f"DEBUG: load_notes called with index: {index}, category_to_select: {category_to_select}")

        self.notes_list_widget.clear() # Not listesini temizle

        try:
            # currentIndexChanged sinyalini geçici olarak kes (sonsuz döngüyü önlemek için)
            self.category_combo_box.currentIndexChanged.disconnect(self.load_notes)
        except TypeError:
            pass # Sinyal zaten bağlı değilse hata verme

        self.category_combo_box.blockSignals(True) # Sinyalleri engelle
        self.category_combo_box.clear() # Kategori ComboBox'ını temizle
        self.category_combo_box.addItem("All Notes") # "Tüm Notlar" seçeneğini ekle
        self.displayed_title_to_note_id = {} # Başlık-ID eşlemesini sıfırla
        self.note_id_to_category = {} # ID-Kategori eşlemesini sıfırla

        # Tüm notların meta verilerini ve tüm kategorileri yükle
        all_notes_metadata, all_categories = note_manager.load_all_notes_metadata(self.db_manager)
        print(f"DEBUG: all_categories: {all_categories}") 

        # Kategorileri ComboBox'a ekle
        for category in sorted(list(all_categories)):
            self.category_combo_box.addItem(category)

        # Seçilecek kategoriyi belirle
        if index is not None and index >= 0 and index < self.category_combo_box.count():
            self.category_combo_box.setCurrentIndex(index)
            selected_category = self.category_combo_box.itemText(index)
        elif category_to_select:
            idx = self.category_combo_box.findText(category_to_select) # Kategori adını bul
            print(f"DEBUG: findText result for {category_to_select}: {idx}") 
            if idx != -1:
                self.category_combo_box.setCurrentIndex(idx)
                selected_category = category_to_select
            else: # Kategori bulunamazsa "Tüm Notlar"ı seç
                self.category_combo_box.setCurrentIndex(0)
                selected_category = "All Notes"
        else: # Varsayılan olarak "Tüm Notlar"ı seç
            self.category_combo_box.setCurrentIndex(0)
            selected_category = "All Notes"

        self.category_combo_box.blockSignals(False) # Sinyalleri tekrar etkinleştir
        self.category_combo_box.currentIndexChanged.connect(self.load_notes) # Sinyali tekrar bağla

        print(f"DEBUG: Selected category: {selected_category}")
        self.note_count_label.setText(f"{selected_category} not sayısı: {self.db_manager.note_count(selected_category)}")
        
        # Notları filtreleyerek listeye ekle
        for note_id, display_title, category_path in all_notes_metadata:
            if category_path:
                self.note_id_to_category[note_id] = category_path # Not ID'si ile kategori eşlemesini sakla

            # Seçili kategoriye veya "Tüm Notlar"a göre filtrele
            if selected_category == "All Notes" or category_path == selected_category:
                self.notes_list_widget.addItem(display_title) # Notu listeye ekle
                self.displayed_title_to_note_id[display_title] = note_id # Başlık-ID eşlemesini sakla

    # open_selected_note metodu, not listesinden seçilen bir notu düzenleyiciye yükler.
    # item: Seçilen QListWidgetItem nesnesi.
    def open_selected_note(self, item):
        log_debug(f"DEBUG: open_selected_note çağrıldı. Seçilen item: {item.text()}")
        selected_display_title = item.text() # Seçilen notun görünen başlığını al
        note_id = self.displayed_title_to_note_id.get(selected_display_title) # Başlıktan not ID'sini al
        note_category = self.note_id_to_category.get(note_id, "") # Notun kategorisini al

        if not note_id: # Not ID'si bulunamazsa hata mesajı göster
            QMessageBox.critical(self, "Error", f"Could not find note ID for note: {selected_display_title}")
            return

        self.current_note_id = note_id # Mevcut not ID'sini güncelle
        self.current_note_category = note_category # Mevcut not kategorisini güncelle
        log_debug(f"DEBUG: open_selected_note - current_note_id atandı: {self.current_note_id}")
        self.display_linked_notes() # Bağlantılı notları her durumda göster

        content = note_manager.get_note_content(self.db_manager, self.current_note_id) # Not içeriğini veritabanından al
        if content is not None: # İçerik varsa
            self.editor.setPlainText(content) # Editöre içeriği yükle
            self.setWindowTitle(f"Zettelkasten AI Notes - {selected_display_title}") # Pencere başlığını güncelle
        else:
            QMessageBox.critical(self, "Error", f"Could not read content for note: {selected_display_title}") # Hata mesajı göster
            log_debug(f"DEBUG: open_selected_note - içerik okunamadı! Note ID: {self.current_note_id}")
            self.new_note() # Yeni boş not oluştur

    # update_preview metodu, düzenleyicideki Markdown metnini HTML'e dönüştürür
    # ve önizleme alanında gösterir.
    def update_preview(self):
        markdown_text = self.editor.toPlainText() # Editördeki metni al
        html = markdown.markdown(markdown_text) # Markdown'ı HTML'e dönüştür
        self.preview.setHtml(html) # Önizleme alanına HTML'i ayarla

    # handle_ai_generation_finished metodu, AI not oluşturma işlemi tamamlandığında çağrılır.
    # generated_notes: Oluşturulan notların listesi.
    def handle_ai_generation_finished(self, generated_notes):
        if self.loading_dialog: # Yükleme iletişim kutusu açıksa kapat
            self.loading_dialog.close()
            self.loading_dialog = None 

        if generated_notes: # Notlar oluşturulduysa
            log_debug("DEBUG: handle_ai_generation_finished: Yapay zeka notları çalışan tarafından işlendi. Kullanıcı arayüzü yenileniyor.")
            QMessageBox.information(self, "AI Note Generation", "Notlar başarıyla oluşturuldu ve kaydedildi!")
            # Notları yeniden yükle ve mevcut kategoriyi seç
            self.load_notes(category_to_select=self.current_note_category)
            self.display_linked_notes() # Oluşturma ve bağlama sonrası bağlantılı notları yenile
        else:
            QMessageBox.warning(self, "AI Note Generation", "Yapay zeka tarafından not oluşturulmadı.")

    # handle_ai_generation_error metodu, AI not oluşturma sırasında bir hata oluştuğunda çağrılır.
    # message: Hata mesajı.
    def handle_ai_generation_error(self, message):
        if self.loading_dialog: # Yükleme iletişim kutusu açıksa kapat
            self.loading_dialog.close()
            self.loading_dialog = None 
        QMessageBox.critical(self, "AI Note Generation Error", f"An error occurred during AI note generation: {message}")

    # open_selected_note_from_link metodu, bağlantılı notlar listesinden bir not seçildiğinde çağrılır.
    # Seçilen bağlantılı notu ana not listesinde bulur ve açar.
    # item: Seçilen QListWidgetItem nesnesi.
    def open_selected_note_from_link(self, item):
        selected_note_id = item.data(Qt.UserRole) # Seçilen notun ID'sini al
        if selected_note_id:
            display_title = None
            # displayed_title_to_note_id sözlüğünde not ID'sine karşılık gelen başlığı bul
            for title, note_id in self.displayed_title_to_note_id.items():
                if note_id == selected_note_id:
                    display_title = title
                    break
            
            if display_title:
                # Ana not listesindeki öğeyi simüle ederek aç
                # Bu, open_selected_note metodunu tetikleyecek ve içeriği yükleyecektir.
                items = self.notes_list_widget.findItems(display_title, Qt.MatchExactly)
                if items:
                    self.notes_list_widget.setCurrentItem(items[0]) # Öğeyi seçili hale getir
                    self.open_selected_note(items[0]) # Notu aç
            else:
                QMessageBox.warning(self, "Open Linked Note", "Could not find the linked note in the main list.")

    # _show_linked_note_context_menu metodu, bağlantılı notlar listesindeki öğelere sağ tıklandığında
    # açılan bağlam menüsünü gösterir.
    # position: Sağ tıklamanın gerçekleştiği konum.
    def _show_linked_note_context_menu(self, position):
        item = self.linked_notes_list_widget.itemAt(position) # Tıklanan konumdaki öğeyi al
        if item:
            self.linked_notes_list_widget.setCurrentItem(item) # Tıklanan öğeyi seçili hale getir
            context_menu = QMenu(self) # Yeni bir bağlam menüsü oluştur
            unlink_action = context_menu.addAction("Unlink Note") # "Bağlantıyı Kaldır" eylemini ekle
            # Menüyü göster ve seçilen eylemi al
            action = context_menu.exec_(self.linked_notes_list_widget.mapToGlobal(position))
            if action == unlink_action:
                self.unlink_note() # Bağlantıyı kaldır metodunu çağır

    # display_linked_notes metodu, mevcut notla bağlantılı notları listeler.
    def display_linked_notes(self):
        self.linked_notes_list_widget.clear() # Bağlantılı notlar listesini temizle
        if self.current_note_id: # Mevcut bir not seçili ise
            linked_note_ids = self.db_manager.get_note_links(self.current_note_id) # Bağlantılı not ID'lerini al
            if linked_note_ids: # Bağlantılı notlar varsa
                for linked_id in linked_note_ids:
                    note_data = self.db_manager.get_note(linked_id) # Bağlantılı notun verilerini al
                    if note_data:
                        linked_title = note_data[1] # Not başlığını al (indeks 1)
                        item = QListWidgetItem(linked_title) # Yeni bir liste öğesi oluştur
                        item.setData(Qt.UserRole, linked_id) # Gerçek not ID'sini öğenin verisine sakla
                        self.linked_notes_list_widget.addItem(item) # Listeye ekle
            else:
                self.linked_notes_list_widget.clear()
                self.linked_notes_list_widget.addItem("No linked notes.") # Bağlantılı not yoksa mesaj göster
        else:
            self.linked_notes_list_widget.clear()
            self.linked_notes_list_widget.addItem("Select a note to see its links.") # Not seçilmemişse mesaj göster

    # unlink_note metodu, seçili bağlantılı notun mevcut notla olan bağlantısını kaldırır.
    def unlink_note(self):
        selected_items = self.linked_notes_list_widget.selectedItems() # Seçili öğeleri al
        if not selected_items: # Hiçbir öğe seçili değilse uyarı ver
            QMessageBox.information(self, "Unlink Note", "Please select a linked note to unlink.")
            return

        linked_note_id_to_unlink = selected_items[0].data(Qt.UserRole) # Bağlantısı kaldırılacak notun ID'sini al
        linked_note_title_to_unlink = selected_items[0].text() # Bağlantısı kaldırılacak notun başlığını al

        if not self.current_note_id: # Mevcut not seçili değilse hata mesajı göster
            QMessageBox.critical(self, "Error", "No current note selected to unlink from.")
            return

        # Kullanıcıya bağlantıyı kaldırma işlemini onaylaması için soru sor
        reply = QMessageBox.question(self, 'Unlink Note', 
                                     f"Are you sure you want to unlink '{linked_note_title_to_unlink}' from the current note?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes: # Kullanıcı onaylarsa
            # Not bağlantısını veritabanından sil
            success = self.db_manager.delete_note_link(self.current_note_id, linked_note_id_to_unlink)
            if success:
                QMessageBox.information(self, "Unlink Note", f"Successfully unlinked '{linked_note_title_to_unlink}'.")
                self.display_linked_notes() # Bağlantılı notlar listesini yenile
            else:
                QMessageBox.critical(self, "Error", f"Failed to unlink note: {linked_note_title_to_unlink}.")


# Uygulama başlangıç noktası
if __name__ == '__main__':
    log_debug("DEBUG: Uygulama başlatıldı ve debug.log dosyası test edildi.")

    app = QApplication(sys.argv) # QApplication nesnesini oluştur
    # Stil dosyasını yükle (varsa)
    stylesheet_path = os.path.join(os.path.dirname(__file__), 'style.qss')
    if os.path.exists(stylesheet_path):
        with open(stylesheet_path, "r") as f:
            app.setStyleSheet(f.read())

    ex = ZettelkastenApp() # Ana uygulama penceresini oluştur
    ex.show() # Pencereyi göster
    sys.exit(app.exec_()) # Uygulamayı çalıştır ve çıkış kodunu döndür
