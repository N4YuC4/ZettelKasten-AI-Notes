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
from uuid import uuid4 # Benzersiz ID oluşturmak için
from datetime import datetime # Zaman damgaları için

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

                notes_to_insert = []
                links_to_insert = []
                new_note_mappings = {}

                # Aşama 1: Not verilerini ve geçici kimlikleri hazırla
                for note_data in generated_notes:
                    temp_id = str(uuid4())
                    title = note_data.get('title', 'Untitled Note')
                    sanitized_title = note_manager.get_sanitized_title(f"# {title}")
                    new_note_mappings[sanitized_title] = temp_id
                    note_data['_temp_id'] = temp_id

                # Hem mevcut hem de yeni not başlıklarını birleştir
                title_to_id.update(new_note_mappings)

                now = datetime.now().isoformat()
                # Aşama 2: Notları ve bağlantıları eklemek için listeleri oluştur
                for note_data in generated_notes:
                    source_id = note_data['_temp_id']
                    title = note_data.get('title', 'Untitled Note')
                    content = note_data.get('content', '')
                    category = note_data.get('general_title', 'AI Generated')
                    
                    full_content = f"# {title}\n\n{content}"
                    sanitized_title = note_manager.get_sanitized_title(full_content)

                    notes_to_insert.append(
                        (source_id, sanitized_title, full_content, category, now, now)
                    )

                    connections = note_data.get('connections', [])
                    for target_title_raw in connections:
                        sanitized_target_title = note_manager.get_sanitized_title(target_title_raw)
                        target_id = title_to_id.get(sanitized_target_title)
                        if target_id:
                            links_to_insert.append((source_id, target_id))
                        else:
                            log_debug(f"DEBUG: Could not find target_id for '{sanitized_target_title}' (raw: '{target_title_raw}'). Link not inserted.")

                # Aşama 3: Toplu veritabanı işlemleri
                if notes_to_insert:
                    db_manager_worker.bulk_insert_notes(notes_to_insert)
                    log_debug(f"DEBUG: Bulk inserted {len(notes_to_insert)} notes.")

                if links_to_insert:
                    db_manager_worker.bulk_insert_links(links_to_insert)
                    log_debug(f"DEBUG: Bulk inserted {len(links_to_insert)} links.")

                self.finished.emit(generated_notes)
            else:
                self.error.emit("AI did not generate any notes from the PDF content.")
        except ValueError as ve:
            self.error.emit(f"API Key Error: {str(ve)}")
        except Exception as e:
            self.error.emit(f"An error occurred during AI note generation: {e}")
        finally:
            if db_manager_worker:
                db_manager_worker.close_connection()
