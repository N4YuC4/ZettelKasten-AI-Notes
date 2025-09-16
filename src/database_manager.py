# database_manager.py
#
# Bu dosya, uygulamanın SQLite veritabanı ile etkileşimini yöneten
# DatabaseManager sınıfını içerir. Notların, kategorilerin ve not bağlantılarının
# CRUD (Oluşturma, Okuma, Güncelleme, Silme) işlemlerini sağlar.

import sqlite3 # SQLite veritabanı ile çalışmak için
import os # Dosya sistemi işlemleri için
from datetime import datetime # Zaman damgaları için

# Veritabanı dosyasının yolu. Uygulama kök dizinindeki 'db' klasöründe bulunur.
DATABASE_FILE = os.path.join("db", "notes.db")

# DatabaseManager sınıfı, SQLite veritabanı bağlantısını ve işlemlerini yönetir.
class DatabaseManager:
    # __init__ metodu, veritabanı bağlantısını kurar ve gerekli tabloları oluşturur.
    def __init__(self):
        # 'db' klasörünün var olduğundan emin ol, yoksa oluştur
        os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
        self.conn = sqlite3.connect(DATABASE_FILE) # Veritabanına bağlan
        self.conn.execute("PRAGMA foreign_keys = ON") # Yabancı anahtar kısıtlamalarını etkinleştir
        self.create_notes_table() # Notlar tablosunu oluştur
        self.create_note_links_table() # Not bağlantıları tablosunu oluştur
        self._create_settings_table() # Ayarlar tablosunu oluştur

    # _create_settings_table metodu, uygulama ayarlarını saklamak için bir tablo oluşturur.
    def _create_settings_table(self):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, # Ayar anahtarı (benzersiz)
                value TEXT # Ayar değeri
            )
        """)
        self.conn.commit() # Değişiklikleri kaydet

    # create_notes_table metodu, not bilgilerini saklamak için bir tablo oluşturur.
    def create_notes_table(self):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY, # Notun benzersiz ID'si
                title TEXT NOT NULL, # Notun başlığı (boş olamaz)
                content TEXT, # Notun içeriği
                category TEXT DEFAULT '', # Notun kategorisi (varsayılan boş)
                created_at TEXT NOT NULL, # Oluşturulma zamanı
                updated_at TEXT NOT NULL # Son güncelleme zamanı
            )
        """)
        self.conn.commit() # Değişiklikleri kaydet

    # create_note_links_table metodu, notlar arasındaki bağlantıları saklamak için bir tablo oluşturur.
    def create_note_links_table(self):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS note_links (
                source_note_id TEXT NOT NULL, # Kaynak notun ID'si
                target_note_id TEXT NOT NULL, # Hedef notun ID'si
                PRIMARY KEY (source_note_id, target_note_id), # İki ID'nin kombinasyonu benzersiz olmalı
                FOREIGN KEY (source_note_id) REFERENCES notes(id) ON DELETE CASCADE, # Kaynak not silinirse bağlantı da silinir
                FOREIGN KEY (target_note_id) REFERENCES notes(id) ON DELETE CASCADE # Hedef not silinirse bağlantı da silinir
            )
        """)
        self.conn.commit() # Değişiklikleri kaydet

    # insert_note_link metodu, iki not arasında bir bağlantı ekler.
    # source_note_id: Bağlantının başladığı notun ID'si.
    # target_note_id: Bağlantının bittiği notun ID'si.
    def insert_note_link(self, source_note_id, target_note_id):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        try:
            cursor.execute("""
                INSERT INTO note_links (source_note_id, target_note_id)
                VALUES (?, ?)
            """, (source_note_id, target_note_id)) # Bağlantıyı ekle
            self.conn.commit() # Değişiklikleri kaydet
            return True # Başarılı olduğunu belirt
        except sqlite3.IntegrityError:
            # Bağlantı zaten varsa (PRIMARY KEY kısıtlaması nedeniyle)
            return False # Başarısız olduğunu belirt

    # get_note_links metodu, belirli bir notla bağlantılı tüm notların ID'lerini döndürür.
    # note_id: Bağlantıları sorgulanacak notun ID'si.
    def get_note_links(self, note_id):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("""
            SELECT target_note_id FROM note_links WHERE source_note_id = ?
            UNION # Hem kaynak hem de hedef olarak bağlantılı notları al
            SELECT source_note_id FROM note_links WHERE target_note_id = ?
        """, (note_id, note_id)) # Sorguyu çalıştır
        return [row[0] for row in cursor.fetchall()] # Sonuçları liste olarak döndür

    # delete_note_link metodu, iki not arasındaki belirli bir bağlantıyı siler.
    # source_note_id: Kaynak notun ID'si.
    # target_note_id: Hedef notun ID'si.
    def delete_note_link(self, source_note_id, target_note_id):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("""
            DELETE FROM note_links WHERE source_note_id = ? AND target_note_id = ?
        """, (source_note_id, target_note_id)) # Bağlantıyı sil
        self.conn.commit() # Değişiklikleri kaydet
        return cursor.rowcount > 0 # Bir bağlantı silindiyse True, aksi halde False döndür

    # get_note_id_by_title metodu, bir notun başlığına göre ID'sini döndürür.
    # title: Aranacak notun başlığı.
    def get_note_id_by_title(self, title):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("SELECT id FROM notes WHERE title = ?", (title,)) # Başlığa göre ID'yi sorgula
        result = cursor.fetchone() # İlk sonucu al
        return result[0] if result else None # Sonuç varsa ID'yi, yoksa None döndür

    # insert_note metodu, veritabanına yeni bir not ekler.
    # note_id: Notun benzersiz ID'si.
    # title: Notun başlığı.
    # content: Notun içeriği.
    # category: Notun kategorisi (isteğe bağlı).
    def insert_note(self, note_id, title, content, category=""):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        now = datetime.now().isoformat() # Mevcut zamanı ISO formatında al
        cursor.execute("""
            INSERT INTO notes (id, title, content, category, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (note_id, title, content, category, now, now)) # Notu ekle
        self.conn.commit() # Değişiklikleri kaydet

    # update_note metodu, mevcut bir notu günceller.
    # note_id: Güncellenecek notun ID'si.
    # title: Yeni başlık.
    # content: Yeni içerik.
    # category: Yeni kategori (isteğe bağlı).
    def update_note(self, note_id, title, content, category=""):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        now = datetime.now().isoformat() # Mevcut zamanı ISO formatında al
        cursor.execute("""
            UPDATE notes
            SET title = ?, content = ?, category = ?, updated_at = ?
            WHERE id = ?
        """, (title, content, category, now, note_id)) # Notu güncelle
        self.conn.commit() # Değişiklikleri kaydet

    # delete_note metodu, belirli bir notu veritabanından siler.
    # note_id: Silinecek notun ID'si.
    def delete_note(self, note_id):
        try:
            cursor = self.conn.cursor() # Veritabanı imlecini al
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,)) # Notu sil
            self.conn.commit() # Değişiklikleri kaydet
            return True  # Başarılı olduğunu belirt
        except sqlite3.Error as e:
            print(f"Database error during note deletion: {e}") # Hata mesajını yazdır
            return False # Başarısız olduğunu belirt
    
    # delete_category metodu, belirli bir kategoriyi ve bu kategoriye ait tüm notları siler.
    # category_name: Silinecek kategorinin adı.
    def delete_category(self, category_name):
        try:
            cursor = self.conn.cursor() # Veritabanı imlecini al
            cursor.execute("DELETE FROM notes WHERE category = ?", (category_name,)) # Kategoriye ait notları sil
            self.conn.commit() # Değişiklikleri kaydet
            return True  # Başarılı olduğunu belirt
        except sqlite3.Error as e:
            print(f"Database error during category deletion: {e}") # Hata mesajını yazdır
            return False  # Başarısız olduğunu belirt

    # get_note metodu, belirli bir notun tüm verilerini (ID, başlık, içerik, kategori) döndürür.
    # note_id: Alınacak notun ID'si.
    def get_note(self, note_id):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("SELECT id, title, content, category FROM notes WHERE id = ?", (note_id,)) # Notu sorgula
        note = cursor.fetchone() # İlk sonucu al
        return note # Not verilerini döndür

    # get_all_notes_metadata metodu, tüm notların meta verilerini (ID, başlık, kategori) ve
    # tüm benzersiz kategori adlarını döndürür.
    def get_all_notes_metadata(self):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("SELECT id, title, category FROM notes") # Notların meta verilerini sorgula
        notes_metadata = cursor.fetchall() # Tüm sonuçları al
        all_categories = set() # Benzersiz kategorileri saklamak için küme
        for note_id, title, category in notes_metadata:
            if category:
                all_categories.add(category) # Kategori varsa kümeye ekle
        return notes_metadata, all_categories # Meta verileri ve kategorileri döndür

    # create_category metodu, mevcut şemada kategoriler notların bir parçası olduğu için
    # bir yer tutucudur. Gelecekte ayrı kategori yönetimi için kullanılabilir.
    def create_category(self, category_name):
        # Şu an için, bu kategoriye sahip bir not kaydedildiğinde kategori örtük olarak
        # oluşturulduğu varsayıldığı için her zaman True döndürür.
        return True

    # read_note_content metodu, belirli bir notun içeriğini döndürür.
    # note_id: İçeriği okunacak notun ID'si.
    def read_note_content(self, note_id):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("SELECT content FROM notes WHERE id = ?", (note_id,)) # İçeriği sorgula
        result = cursor.fetchone() # İlk sonucu al
        return result[0] if result else None # Sonuç varsa içeriği, yoksa None döndür

    # save_note metodu, bir notu kaydeder (yeni oluşturur veya günceller).
    # note_id: Güncellenecek notun ID'si (None ise yeni not oluşturulur).
    # note_content: Notun içeriği.
    # category: Notun kategorisi (isteğe bağlı).
    def save_note(self, note_id, note_content, category=""):
        from uuid import uuid4 # Benzersiz ID oluşturmak için
        now = datetime.now().isoformat() # Mevcut zamanı ISO formatında al
        title = note_content.split('\n')[0].strip() # İlk satırı başlık olarak al

        if not title:
            title = "Untitled Note" # Başlık boşsa varsayılan başlık ata

        if note_id:
            # Mevcut notu güncelle
            cursor = self.conn.cursor() # Veritabanı imlecini al
            cursor.execute("UPDATE notes SET title = ?, content = ?, category = ?, updated_at = ? WHERE id = ?",
                           (title, note_content, category, now, note_id)) # Notu güncelle
            self.conn.commit() # Değişiklikleri kaydet
            return note_id, title # Not ID'si ve başlığı döndür
        else:
            # Yeni not oluştur
            new_note_id = str(uuid4()) # Yeni benzersiz ID oluştur
            cursor = self.conn.cursor() # Veritabanı imlecini al
            cursor.execute("INSERT INTO notes (id, title, content, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                           (new_note_id, title, note_content, category, now, now)) # Yeni notu ekle
            self.conn.commit() # Değişiklikleri kaydet
            return new_note_id, title # Yeni not ID'si ve başlığı döndür

    # rename_note metodu, bir notun başlığını günceller.
    # note_id: Yeniden adlandırılacak notun ID'si.
    # new_title: Notun yeni başlığı.
    # category: Notun kategorisi (şu an kullanılmıyor ama uyumluluk için tutuluyor).
    def rename_note(self, note_id, new_title, category=""):
        now = datetime.now().isoformat() # Mevcut zamanı ISO formatında al
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("UPDATE notes SET title = ?, updated_at = ? WHERE id = ?",
                       (new_title, now, note_id)) # Notun başlığını güncelle
        self.conn.commit() # Değişiklikleri kaydet
        return True, new_title # Başarılı olduğunu ve yeni başlığı döndür

    # get_all_note_titles_and_ids metodu, tüm notların başlıklarını ve ID'lerini bir sözlük olarak döndürür.
    def get_all_note_titles_and_ids(self):
        cursor = self.conn.cursor() # Veritabanı imlecini al
        cursor.execute("SELECT title, id FROM notes") # Başlık ve ID'leri sorgula
        return {title: note_id for title, note_id in cursor.fetchall()} # Sözlük olarak döndür

    # close_connection metodu, veritabanı bağlantısını kapatır.
    def close_connection(self):
        if self.conn:
            self.conn.close() # Bağlantıyı kapat
