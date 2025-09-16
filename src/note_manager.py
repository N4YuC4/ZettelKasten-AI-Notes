# note_manager.py
#
# Bu dosya, Zettelkasten notlarının yönetimiyle ilgili yardımcı işlevleri içerir.
# Not başlıklarını temizleme, notları kaydetme, silme, yeniden adlandırma ve
# not meta verilerini yükleme gibi işlemleri gerçekleştirir.

import re # Düzenli ifadeler için
import uuid # Benzersiz ID'ler oluşturmak için
from logger import log_debug # Hata ayıklama loglama fonksiyonu için

# generate_unique_id fonksiyonu, yeni notlar için benzersiz bir ID oluşturur.
def generate_unique_id():
    return str(uuid.uuid4()) # UUID (Universally Unique Identifier) oluştur ve stringe çevir

# get_sanitized_title fonksiyonu, not içeriğinin ilk satırından temizlenmiş bir başlık çıkarır.
# Markdown başlıklarını, parantez içindeki içerikleri ve diğer Markdown formatlama karakterlerini kaldırır.
def get_sanitized_title(content):
    first_line = content.split('\n')[0].strip() # İçeriğin ilk satırını al ve boşlukları temizle
    if not first_line:
        return "Untitled Note" # İlk satır boşsa varsayılan başlık döndür

    # 1. Markdown başlık sözdizimini (örn. #, ##, vb.) satırın başından kaldır
    sanitized_title = re.sub(r'^#+\s*', '', first_line)

    # Remove bold/italic markers
    sanitized_title = re.sub(r'(\*\*|__|\*|_)', '', sanitized_title)
    # Remove inline code markers
    sanitized_title = re.sub(r'`', '', sanitized_title)
    # Remove strikethrough markers
    sanitized_title = re.sub(r'~~', '', sanitized_title)
    # Remove image/link syntax (e.g., ![alt](url) or [text](url))
    sanitized_title = re.sub(r'!\\[.*?]\(.*?\\]\)', '', sanitized_title)
    sanitized_title = re.sub(r'\[.*?]\[.*?]', '', sanitized_title)
    # Remove remaining special characters that might be part of markdown or problematic in titles
    sanitized_title = re.sub(r'[<>:"/\\|?*]', '', sanitized_title)
    
    # Temizleme işleminden sonra oluşabilecek baştaki/sondaki boşlukları kaldır
    sanitized_title = sanitized_title.strip()

    return sanitized_title

# save_note fonksiyonu, bir notu veritabanına kaydeder veya günceller.
# db_manager: Veritabanı yöneticisi nesnesi.
# note_id: Güncellenecek notun ID'si (None ise yeni not oluşturulur).
# note_content: Notun içeriği.
# category_path: Notun kategorisi.
def save_note(db_manager, note_id, note_content, category_path=""):
    sanitized_title = get_sanitized_title(note_content) # İçerikten temizlenmiş başlığı al
    
    if not sanitized_title: # Temizleme sonrası başlık boşsa
        sanitized_title = "Untitled Note" # Varsayılan başlık ata

    # Bu başlığa sahip mevcut bir not var mı kontrol et
    existing_note_id = db_manager.get_note_id_by_title(sanitized_title) # Existing note ID by title

    if existing_note_id: # Mevcut bir not bulunursa
        log_debug(f"DEBUG: Found existing note with sanitized title '{sanitized_title}' and ID '{existing_note_id}'. Updating instead of creating new.")
        # Mevcut notu güncelle
        db_manager.update_note(existing_note_id, sanitized_title, note_content, category_path)
        return existing_note_id, sanitized_title # Mevcut not ID'si ve başlığı döndür
    elif note_id is None: # Mevcut not yoksa ve ID sağlanmamışsa, yeni bir not oluştur
        new_note_id = generate_unique_id() # Yeni benzersiz ID oluştur
        db_manager.insert_note(new_note_id, sanitized_title, note_content, category_path) # Yeni notu ekle
        return new_note_id, sanitized_title # Yeni not ID'si ve başlığı döndür
    else:
        # note_id sağlanmışsa, belirli notu güncelle (bu yol, mevcut notların manuel düzenlemeleri içindir)
        db_manager.update_note(note_id, sanitized_title, note_content, category_path)
        return note_id, sanitized_title # Güncellenen not ID'si ve başlığı döndür

# delete_note fonksiyonu, belirli bir notu veritabanından siler.
# db_manager: Veritabanı yöneticisi nesnesi.
# note_id: Silinecek notun ID'si.
def delete_note(db_manager, note_id):
    try:
        db_manager.delete_note(note_id) # Veritabanından notu sil
        log_debug(f"Note with ID {note_id} deleted from database.")
        return True # Başarılı olduğunu belirt
    except Exception as e:
        log_debug(f"Error deleting note from database: {e}") # Hata mesajını logla
        return False # Başarısız olduğunu belirt

# rename_note fonksiyonu, belirli bir notu yeniden adlandırır.
# db_manager: Veritabanı yöneticisi nesnesi.
# note_id: Yeniden adlandırılacak notun ID'si.
# new_title: Notun yeni başlığı.
# category_path: Notun kategorisi (şu an kullanılmıyor ama uyumluluk için tutuluyor).
def rename_note(db_manager, note_id, new_title, category_path=""):
    sanitized_new_title = get_sanitized_title(new_title) # Yeni başlığı temizle
    if not sanitized_new_title:
        return False, "New title cannot be empty or result in an empty sanitized title."

    note_data = db_manager.get_note(note_id) # Notun mevcut verilerini al
    if note_data:
        current_content = note_data[2] # Notun içeriği (indeks 2)
        # Notu yeni başlık ve mevcut içerikle güncelle
        db_manager.update_note(note_id, sanitized_new_title, current_content, category_path)
        log_debug(f"Note with ID {note_id} renamed to {sanitized_new_title}.")
        return True, sanitized_new_title # Başarılı olduğunu ve yeni başlığı döndür
    else:
        return False, "Note not found."

# load_all_notes_metadata fonksiyonu, tüm notların meta verilerini ve tüm kategorileri yükler.
# db_manager: Veritabanı yöneticisi nesnesi.
def load_all_notes_metadata(db_manager):
    notes_metadata_from_db, all_categories_from_db = db_manager.get_all_notes_metadata() # Veritabanından meta verileri ve kategorileri al
    
    notes_metadata = [] # (display_title, note_id, category_path) formatında not meta verileri listesi
    all_categories = set() # Benzersiz kategorileri saklamak için küme

    for note_id, title, category in notes_metadata_from_db:
        notes_metadata.append((note_id, title, category)) # Meta verileri listeye ekle
        if category:
            all_categories.add(category) # Kategori varsa kümeye ekle
            
    return notes_metadata, sorted(list(all_categories_from_db)) # Meta verileri ve sıralanmış kategorileri döndür

# get_note_content fonksiyonu, belirli bir notun içeriğini döndürür.
# db_manager: Veritabanı yöneticisi nesnesi.
# note_id: İçeriği alınacak notun ID'si.
def get_note_content(db_manager, note_id):
    note_data = db_manager.get_note(note_id) # Not verilerini al
    if note_data:
        return note_data[2] # İçeriği döndür (indeks 2)
    return None # Not bulunamazsa None döndür

# create_category fonksiyonu, kategori oluşturma işlemini yönetir.
# Mevcut veritabanı şemasında kategoriler notun bir alanı olduğu için
# bu fonksiyon sadece kategori adının geçerli olup olmadığını kontrol eder.
# category_name: Oluşturulacak kategorinin adı.
def create_category(category_name):
    # Veritabanı depolamasında, kategoriler sadece nottaki bir alandır.
    # Fiziksel bir dizin oluşturmaya gerek yoktur.
    # Bu fonksiyon, kategori adı geçerliyse True döndürebilir.
    if category_name.strip(): # Kategori adı boşluklardan sonra boş değilse
        return True
    return False