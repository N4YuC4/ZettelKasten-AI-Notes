# gemini_api_client.py
#
# Bu dosya, Google Gemini API ile etkileşim kurarak Zettelkasten tarzı notlar
# oluşturan GeminiApiClient sınıfını içerir. Verilen metin içeriğinden anahtar
# kavramları, argümanları ve içgörüleri çıkarır ve bunları yapılandırılmış
# JSON formatında notlara dönüştürür.

import google.generativeai as genai # Google Gemini API kütüphanesi
from dotenv import load_dotenv # .env dosyasından ortam değişkenlerini yüklemek için
import os # Ortam değişkenlerine erişim için
import json # JSON verilerini işlemek için
from database_manager import DatabaseManager # Veritabanı yöneticisi (şu an doğrudan kullanılmıyor ama bağımlılık olabilir)
from logger import log_debug # Hata ayıklama loglama fonksiyonu için

# GeminiApiClient sınıfı, Gemini API ile iletişim kurar ve not oluşturma isteklerini yönetir.
class GeminiApiClient:
    # __init__ metodu, API istemcisini başlatır, API anahtarını yükler ve Gemini modelini yapılandırır.
    def __init__(self):
        load_dotenv() # .env dosyasındaki ortam değişkenlerini yükle
        api_key = os.getenv("GEMINI_API_KEY") # Ortam değişkenlerinden API anahtarını al

        if not api_key: # API anahtarı bulunamazsa hata fırlat
            raise ValueError("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable in a .env file or via the application's menu (Settings -> Enter Gemini API Key).")

        genai.configure(api_key=api_key) # Gemini API'yi anahtar ile yapılandır
        self.model = genai.GenerativeModel('gemini-2.5-flash') # Kullanılacak Gemini modelini belirt

    # generate_zettelkasten_notes metodu, verilen metin içeriğinden Zettelkasten tarzı notlar oluşturur.
    # text_content (str): Notların oluşturulacağı metin içeriği.
    # Dönüş: Oluşturulan notların bir listesi. Her not, 'general_title', 'title', 'content' ve 'connections'
    # anahtarlarına sahip bir sözlüktür.
    def generate_zettelkasten_notes(self, text_content):
        """
        Generates Zettelkasten-style notes from the given text content using the Gemini API.
        Args:
            text_content (str): The text content from which to generate notes.
        Returns:
            list: A list of generated notes, where each note is a dictionary
                  with 'general_title', 'title', 'content' and 'connections' keys.
        """
        # Gemini API'ye gönderilecek istem (prompt) metni
        prompt = f"""You are an AI assistant specialized in generating Zettelkasten-style notes. all notes should be concise, focused on a single idea, and formatted in markdown. Make sure markdown formatting is perfectly correct.     
The notes language should be what documents language is.
From the following text, extract key concepts, arguments, and insights.
For each insight, create a concise Zettelkasten note. Each note should be self-contained and atomic.

Format your output as a JSON array of objects, where each object has a 'general_title', 'title', 'content' and 'connections' key.

The 'general_title' should be a single title about what all notes talking about.

Aim to create a rich network of *exceptionally relevant and semantically robust* interconnected notes. For every single connection, there MUST be a crystal-clear, undeniable conceptual link between the core ideas of the two notes. Connections should represent relationships such as:
-   **Elaboration/Detail:** One note provides essential, deeper insight or specific examples for a concept introduced in another.
-   **Causality (Cause/Effect):** One note directly causes or is a direct consequence of Note B.
-   **Dependency (Prerequisite/Follow-up):** Understanding Note A is absolutely necessary to grasp Note B, or Note B logically extends from Note A.
-   **Direct Comparison/Contrast:** Notes directly compare or highlight fundamental differences between specific concepts.
-   **Concrete Example/Direct Application:** One note provides a specific, illustrative example or a practical application of a principle in Note B.
-   **Direct Contradiction/Alternative Perspective:** Notes present opposing, mutually exclusive, or significantly different viewpoints on the SAME specific concept.
Strictly avoid connections based on mere keyword overlap, general thematic association, or weak inferential links. If a note does not have an *unquestionably strong, direct, and conceptually indispensable* connection to another note within the generated set, you MUST NOT create a connection. Prioritize the highest possible quality and precision of connections, even if it means fewer links. **If there is any doubt about the strength or directness of a connection, DO NOT create it.** Focus on creating only the most essential and undeniable links.

Each 'connections' key must contain a list of *exact titles* of other related notes that are also being generated in this JSON array. These titles must precisely match the 'title' field of the notes you generate. If a note has no *strong and direct* relevant connections within the generated set, this list can be empty.

Example:
[
  {{"general_title": "Zettelkasten Method", "title": "Concept of Zettelkasten", "content": "Zettelkasten is a personal knowledge management and note-taking method used in research and study. It consists of individual notes with unique IDs, interconnected by links.", "connections": []}},
  {{"general_title": "Zettelkasten Method", "title": "Atomic Notes Principle", "content": "Each Zettelkasten note should contain only one idea or concept to ensure atomicity and reusability.", "connections": []}}
]

Text to process:
{text_content}
"""
        try:
            response = self.model.generate_content(prompt) # Gemini API'ye istek gönder
            notes_json_str = response.text # API yanıtını metin olarak al
            log_debug(f"DEBUG: Raw Gemini API response (full, len={len(notes_json_str)}): {notes_json_str}")
            
            try:
                notes = json.loads(notes_json_str) # JSON metnini ayrıştır
            except json.JSONDecodeError:
                # Doğrudan ayrıştırma başarısız olursa, Markdown kod bloğu sınırlayıcılarını kaldırmayı dene
                if notes_json_str.startswith('```json') and notes_json_str.endswith('```'):
                    notes_json_str = notes_json_str[len('```json\n'):-len('\n```')] # Sınırlayıcıları kaldır
                    notes = json.loads(notes_json_str) # Tekrar ayrıştırmayı dene
                else:
                    raise # Sınırlayıcıları kaldırmak işe yaramazsa hatayı tekrar fırlat

            return notes # Ayrıştırılan notları döndür
        except json.JSONDecodeError as jde:
            log_debug(f"Error parsing JSON from Gemini API: {jde}") # JSON ayrıştırma hatasını logla
            return [] # Boş liste döndür
        except Exception as e:
            log_debug(f"Error generating notes with Gemini API: {e}") # Diğer hataları logla
            return [] # Boş liste döndür

# Bu blok, dosya doğrudan çalıştırıldığında örnek kullanım sağlar (test amaçlı).
if __name__ == '__main__':
    # Örnek kullanım (test amaçlı)
    # GEMINI_API_KEY ortam değişkenini ayarlamanız gerekmektedir.
    # Örneğin: export GEMINI_API_KEY="YOUR_API_KEY"
    # client = GeminiApiClient()
    # dummy_text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    # generated_notes = client.generate_zettelkasten_notes(dummy_text)
    # if generated_notes:
    #     log_debug("Generated Notes:")
    #     for note in generated_notes:
    #         log_debug(f"Title: {note['title']}")
    #         log_debug(f"Content: {note['content']}")
    #         log_debug("---")
    pass
