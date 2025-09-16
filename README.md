# Zettelkasten AI Notları

![Python Sürümü](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.x-green.svg)
![Lisans](https://img.shields.io/badge/License-MIT-yellow.svg)

## Proje Açıklaması
Zettelkasten AI Notları, Zettelkasten yöntemini kullanarak verimli bilgi yönetimi için tasarlanmış, PyQt5 ile oluşturulmuş bir masaüstü uygulamasıdır. Kullanıcıların notları oluşturmasına, düzenlemesine ve bağlamasına olanak tanır; zengin içerik için Markdown'ı destekler ve gerçek zamanlı önizleme sunar. Benzersiz bir özelliği, Google Gemini AI'yı kullanarak PDF belgelerinden Zettelkasten tarzı notlar oluşturabilmesi ve notlar arasında otomatik olarak ilgili bağlantılar önermesidir. Tüm notlar ve meta verileri yerel bir SQLite veritabanında kalıcı olarak saklanır.

## Temel Özellikler
*   **Sezgisel Kullanıcı Arayüzü:** Sorunsuz not yönetimi için temiz ve duyarlı bir GUI.
*   **Not Oluşturma ve Yönetimi:** Bireysel notları kolayca oluşturun, kaydedin, yeniden adlandırın ve silin.
*   **Canlı Önizlemeli Markdown Desteği:** Notlarınızı Markdown'da yazın ve oluşturulan çıktıyı gerçek zamanlı olarak görün.
*   **Kategorizasyon:** Notları özel kategorilere ayırarak daha iyi filtreleme ve gezinme sağlayın.
*   **PDF'ten AI Destekli Not Oluşturma:** PDF belgelerinden metin çıkarın ve Google Gemini AI'yı kullanarak otomatik olarak Zettelkasten tarzı notlar oluşturun, ilgili bağlantılar önerin.
*   **Akıllı Not Bağlantısı:** Notlar arasında açık bağlantılar oluşturarak zengin, birbirine bağlı bir bilgi grafiği oluşturun.
*   **SQLite Veritabanı:** Tüm notlarınız ve ilişkileri için sağlam ve güvenilir yerel depolama.

## Ekran Görüntüleri
*Yakında...*

## Kurulum

### Önkoşullar
*   Python 3.8 veya üzeri
*   `pip` (Python paket yükleyicisi)

### Adımlar

1.  **Depoyu klonlayın:**
    ```bash
    git clone https://github.com/your-username/Zettelkasten-AI-Notes.git
    cd Zettelkasten-AI-Notes
    ```

2.  **Sanal bir ortam oluşturun ve etkinleştirin (önerilir):**
    ```bash
    python -m venv venv
    # Windows'ta:
    .\venv\Scripts\activate
    # macOS/Linux'ta:
    source venv/bin/activate
    ```

3.  **Bağımlılıkları yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Not: Eğer `requirements.txt` mevcut değilse, `PyQt5`, `python-dotenv`, `markdown`, `google-generativeai`, `PyPDF2`, `uuid` gibi gerekli paketleri yükledikten sonra `pip freeze > requirements.txt` komutunu çalıştırarak oluşturmanız gerekebilir.)*

4.  **Google Gemini API Anahtarınızı ayarlayın:**
    Uygulama, AI notları oluşturmak için bir Google Gemini API Anahtarı gerektirir.
    *   API anahtarınızı [Google AI Studio](https://aistudio.google.com/app/apikey) adresinden edinin.
    *   Projenin kök dizininde (örn. `Zettelkasten-AI-Notes/.env`) `.env` adında bir dosya oluşturun.
    *   API anahtarınızı bu dosyaya aşağıdaki formatta ekleyin:
        ```
        GEMINI_API_KEY="BURAYA_API_ANAHTARINIZI_GIRIN"
        ```
    *   Alternatif olarak, API anahtarını doğrudan uygulama içinden `Ayarlar -> Gemini API Anahtarını Girin` menü seçeneği aracılığıyla girebilirsiniz.

## Kullanım

1.  **Uygulamayı başlatın:**
    ```bash
    python src/main.py
    ```

2.  **Temel Not Yönetimi:**
    *   **Yeni Not:** Düzenleyiciyi temizlemek ve yeni bir nota başlamak için "Yeni Not" düğmesine tıklayın.
    *   **Notu Kaydet:** Düzenleyiciye Markdown içeriğinizi yazın. İlk satır otomatik olarak notun başlığı olacaktır. Veritabanına kaydetmek için "Notu Kaydet" düğmesine tıklayın.
    *   **Notu Yeniden Adlandır:** Listeden bir not seçin, ardından başlığını değiştirmek için "Notu Yeniden Adlandır" düğmesine tıklayın.
    *   **Notu Sil:** Bir not seçin ve kaldırmak için "Notu Sil" düğmesine tıklayın.

3.  **Kategori Yönetimi:**
    *   **Yeni Kategori Oluştur:** "Yeni Kategori"ye tıklayın, bir ad girin ve o kategori içinde yeni bir not oluşturulacaktır.
    *   **Kategoriye Göre Filtrele:** Belirli bir kategoriye veya "Tüm Notlar"a ait notları görüntülemek için kategori açılır menüsünü kullanın.
    *   **Kategoriyi Sil:** Açılır menüden bir kategori seçin ve "Kategoriyi Sil" düğmesine tıklayın. Bu, kategoriyi ve onunla ilişkili tüm notları silecektir.

4.  **Markdown Önizleme:**
    Sol düzenleyici bölmesinde yazdıkça, sağ bölme notunuzun canlı Markdown önizlemesini gösterecektir.

5.  **PDF'ten AI Not Oluşturma:**
    *   "PDF'ten AI Notları Oluştur" düğmesine tıklayın.
    *   Sisteminizden bir PDF dosyası seçin.
    *   Uygulama, PDF'ten metin çıkaracak ve not oluşturma için Gemini AI'ya gönderecektir.
    *   Oluşturulan notlar otomatik olarak kaydedilecek ve not listenize "AI Tarafından Oluşturuldu" veya AI tarafından sağlanan genel bir başlık altında eklenecektir.

6.  **Not Bağlantısı:**
    *   **Notları Bağla:** Ana listeden bir not seçin. "Notu Bağla" düğmesine tıklayın veya nota sağ tıklayıp "Bağla..." seçeneğini seçin. Başka bir notu aramanıza ve seçmenize olanak tanıyan bir iletişim kutusu görünecektir.
    *   **Bağlantılı Notları Görüntüle:** Ana listede bir not seçildiğinde, bağlantılı notları "Bağlantılı Notlar Listesi" bölmesinde görünecektir.
    *   **Bağlantılı Notu Aç:** "Bağlantılı Notlar Listesi"ndeki bağlantılı bir nota tıklayarak düzenleyicide açın.
    *   **Bağlantıyı Kaldır:** "Bağlantılı Notlar Listesi"ndeki bir nota sağ tıklayın ve bağlantıyı kaldırmak için "Bağlantıyı Kaldır" seçeneğini seçin.

## Proje Yapısı
```
Zettelkasten-AI-Notes/
├── src/
│   ├── ai_note_generator_worker.py # AI not oluşturmayı ayrı bir iş parçacığında yönetir
│   ├── database_manager.py         # Tüm SQLite veritabanı işlemlerini yönetir
│   ├── gemini_api_client.py        # Google Gemini API ile arayüz oluşturur
│   ├── main.py                     # Ana uygulama giriş noktası ve GUI mantığı
│   ├── note_manager.py             # Notla ilgili işlemleri yönetir (kaydet, yeniden adlandır, sil, temizle)
│   ├── pdf_processor.py            # PDF dosyalarından metin içeriğini çıkarır
│   └── style.qss                   # Uygulama stil sayfası (PyQt5 stil için QSS)
├── LICENSE                  # Proje lisansı
├── README.md                # Proje genel bakışı ve kurulum talimatları
└── requirements.txt         # Python bağımlılıkları
```

## Katkıda Bulunma
Katkılar memnuniyetle karşılanır! İyileştirmeler, hata düzeltmeleri veya yeni özellikler için önerileriniz varsa, lütfen çekinmeyin:
1.  Depoyu Fork'layın.
2.  Yeni bir dal oluşturun (`git checkout -b feature/YeniOzellik`).
3.  Değişikliklerinizi yapın.
4.  Değişikliklerinizi commit'leyin (`git commit -m 'Yeni bir özellik ekle'`).
5.  Dala push yapın (`git push origin feature/YeniOzellik`).
6.  Bir Pull Request açın.

## Lisans
Bu proje MIT Lisansı altında lisanslanmıştır - ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

## İletişim
Herhangi bir soru veya geri bildirim için lütfen GitHub deposunda bir sorun (issue) açın.