# ai_note_generator_worker.py
#
# Bu dosya, yapay zeka (AI) kullanarak PDF içeriğinden Zettelkasten tarzı notlar oluşturan
# ve bu notları veritabanına kaydeden bir işçi sınıfını (worker class) tanımlar.
# İşlem, GUI'yi dondurmamak için ayrı bir iş parçacığında (thread) yürütülür.

from PyQt5.QtCore import QObject, pyqtSignal # PyQt sinyal ve nesne sistemi için
from gemini_api_client import GeminiApiClient # Gemini API ile etkileşim için
import note_manager # Not yönetimi işlevleri için (kaydetme, başlık sanitizasyonu)
import database_manager # Veritabanı işlemleri için
from logger import log_debug # Hata ayıklama loglama fonksiyonu için

# AiNoteGeneratorWorker sınıfı, AI not oluşturma işlemini ayrı bir iş parçacığında yürütür.
# QObject'ten türetilmiştir, böylece sinyal/slot mekanizmasını kullanabilir.
class AiNoteGeneratorWorker(QObject):
    # finished sinyali: İşlem başarıyla tamamlandığında oluşturulan notların listesini yayar.
    finished = pyqtSignal(list) 
    # error sinyali: İşlem sırasında bir hata oluştuğunda hata mesajını yayar.
    error = pyqtSignal(str) 

    # __init__ metodu, işçi nesnesini başlatır.
    # extracted_text: PDF'ten çıkarılan metin.
    def __init__(self, extracted_text):
        super().__init__()
        self.extracted_text = extracted_text # İşlenecek metni sakla

    # run metodu, iş parçacığı başladığında çağrılan ana işlevdir.
    # AI not oluşturma, kaydetme ve bağlama mantığını içerir.
    def run(self):
        # Bu iş parçacığı için yeni bir DatabaseManager örneği oluştur.
        # Her iş parçacığının kendi veritabanı bağlantısı olmalıdır.
        db_manager_worker = database_manager.DatabaseManager()
        try:
            gemini_client = GeminiApiClient() # Gemini API istemcisini oluştur
            # Gemini API'yi kullanarak Zettelkasten notları oluştur
            generated_notes = gemini_client.generate_zettelkasten_notes(self.extracted_text)

            if generated_notes: # Notlar başarıyla oluşturulduysa
                # Kapsamlı bir arama için mevcut notları ve ID'lerini yükle
                title_to_id = db_manager_worker.get_all_note_titles_and_ids()
                log_debug(f"DEBUG: Initial title_to_id: {title_to_id}")
                note_ids = [] # Yeni kaydedilen notların ID'lerini saklamak için liste
                notes_saved_count = 0 # Kaydedilen not sayacı

                # Aşama 1: Tüm notları kaydet ve title_to_id sözlüğünü yeni kaydedilen notlarla güncelle
                for note_data in generated_notes:
                    title = note_data.get('title', 'Untitled Note') # Not başlığını al
                    content = note_data.get('content', '') # Not içeriğini al
                    category = note_data.get('general_title', 'AI Generated') # Not kategorisini al
                    if title and content: # Başlık ve içerik boş değilse
                        # Notu kaydet ve yeni notun ID'si ile başlığını al
                        new_note_id, saved_title = note_manager.save_note(db_manager_worker, None, f"# {title}\n\n{content}", category)
                        title_to_id[saved_title] = new_note_id # Yeni notu title_to_id sözlüğüne ekle/güncelle
                        note_ids.append(new_note_id) # Not ID'sini listeye ekle
                        notes_saved_count += 1 # Sayacı artır
                        log_debug(f"DEBUG: Saved note '{saved_title}' with ID '{new_note_id}'")
                log_debug(f"DEBUG: Final title_to_id after new notes: {title_to_id}")

                # Aşama 2: Tüm notlar kaydedildikten sonra bağlantıları ekle
                for i, note_data in enumerate(generated_notes):
                    source_id = note_ids[i] # Kaynak notun ID'si
                    connections = note_data.get('connections', []) # Bağlantıları al
                    for target_title_raw in connections:
                        # Hedef başlığı sanitize et (temizle)
                        sanitized_target_title = note_manager.get_sanitized_title(target_title_raw)
                        log_debug(f"DEBUG: Looking for target_title {repr(sanitized_target_title)} (len: {len(sanitized_target_title)}, raw: {repr(target_title_raw)}) in title_to_id.")
                        target_id = title_to_id.get(sanitized_target_title) # Hedef notun ID'sini bul
                        if target_id: # Hedef not bulunduysa
                            log_debug(f"DEBUG: Found target_id '{target_id}' for '{sanitized_target_title}'. Inserting link from '{source_id}' to '{target_id}'.")
                            log_debug(f"DEBUG: Attempting to insert link: Source ID: {source_id}, Target ID: {target_id}, Raw Target Title: {target_title_raw}")
                            db_manager_worker.insert_note_link(source_id, target_id) # Bağlantıyı veritabanına ekle
                        else:
                            log_debug(f"DEBUG: Could not find target_id for '{sanitized_target_title}' (raw: '{target_title_raw}'). Link not inserted.")
                self.finished.emit(generated_notes) # Başarıyla tamamlandığını bildir ve notları yay
            else:
                self.error.emit("AI did not generate any notes from the PDF content.") # Not oluşturulmadıysa hata yay
        except ValueError as ve:
            self.error.emit(f"API Key Error: {str(ve)}") # API anahtarı hatası yay
        except Exception as e:
            self.error.emit(f"An error occurred during AI note generation: {e}") # Diğer hataları yay
        finally:
            # İşçinin veritabanı bağlantısını kapat
            db_manager_worker.close_connection()
