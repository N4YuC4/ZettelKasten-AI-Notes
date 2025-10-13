# Zettelkasten AI Notes

![Python Sürümü](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.x-green.svg)
![Lisans](https://img.shields.io/badge/License-MIT-yellow.svg)

Zettelkasten yöntemini kullanarak verimli bilgi yönetimi için `PyQt5` ile oluşturulmuş bir masaüstü uygulamasıdır. Kullanıcıların notları oluşturmasına, düzenlemesine ve bağlamasına olanak tanır, zengin içerik için `Markdown`'ı destekler ve gerçek zamanlı bir önizleme sunar. Tüm verileri yerel bir `SQLite` veritabanında saklar. Önemli bir özelliği, PDF belgelerinden Zettelkasten tarzı notlar oluşturmak için `Google Gemini AI`'ı kullanabilmesi ve aralarında otomatik olarak ilgili bağlantılar önermesidir. Bu proje, Pupilica tarafından düzenlenen Yapay Zeka Hackathonu için geliştirilmiştir.

## Key Features (Temel Özellikler)

*   **Intuitive User Interface:** Sorunsuz not yönetimi için temiz ve duyarlı bir `GUI`.
*   **Note Creation and Management:** Bireysel notları kolayca oluşturun, kaydedin, yeniden adlandırın ve silin.
*   **Markdown Support with Live Preview:** Notlarınızı `Markdown`'da yazın ve oluşturulan çıktıyı gerçek zamanlı olarak görün.
*   **Categorization:** Daha iyi filtreleme ve gezinme için notları özel kategorilere ayırın.
*   **AI-Powered Note Generation from PDF:** PDF belgelerinden metin çıkarın ve önerilen ilgili bağlantılarla otomatik olarak Zettelkasten tarzı notlar oluşturmak için `Google Gemini AI`'ı kullanın.
*   **Mind Map Visualization:** Notlarınızı ve bağlantılarını etkileşimli bir zihin haritası olarak görüntüleyin. Bu, fikirleriniz arasındaki ilişkileri görselleştirmenize ve bilgi tabanınızda daha etkili bir şekilde gezinmenize yardımcı olur.
*   **Smart Note Linking:** Bilginizin zengin, birbirine bağlı bir grafiğini oluşturmak için notlar arasında açık bağlantılar oluşturun.
*   **Theming:** Uygulamanın görünümünü özelleştirmek için aydınlık ve karanlık temalar arasında seçim yapın.
*   **SQLite Database:** Tüm notlarınız ve ilişkileri için sağlam ve güvenilir yerel depolama.

## How it Works (Nasıl Çalışır?)

`main.py` dosyası, `PyQt5` uygulamasını başlatır ve ana pencereyi (`ZettelkastenApp`) kurar. `database_manager.py` aracılığıyla `SQLite` veritabanına bağlanır. Bir kullanıcı uygulamayla etkileşimde bulunduğunda (ör. düzenleyiciye yazar, `save`'e tıklar, bir kategori seçer), `ZettelkastenApp` olayları yönetir, `UI`'ı günceller ve veritabanındaki notlar üzerinde `CRUD` (Create, Read, Update, Delete) işlemlerini gerçekleştirmek için `database_manager.py`'deki uygun yöntemleri çağırır. Bir notun içeriğinin ilk satırı otomatik olarak başlığı olarak kullanılır.

Yapay zeka notu oluşturma, `UI`'ı dondurmamak için ayrı bir `thread`'de çalışan `ai_note_generator_worker.py` tarafından yönetilir. Bir PDF'den çıkarılan metni `gemini_api_client.py` aracılığıyla `Google Gemini API`'ye gönderir ve ardından notları oluşturmak ve bağlamak için yanıtı işler.

Zihin haritası görselleştirmesi, notları ve bağlantılarını görüntülemek için bir `force-directed graph layout` kullanan `mind_map_widget.py` tarafından yönetilir.

## Ekran Görüntüleri
*Yakında...*

## Installation (Kurulum)

### Önkoşullar
*   Python 3.8 veya üstü
*   `pip` (Python paket yükleyici)

### Adımlar

1.  **Depoyu klonlayın:**
    ```bash
    git clone https://github.com/kullanici-adiniz/Zettelkasten-AI-Notes.git
    cd Zettelkasten-AI-Notes
    ```

2.  **Sanal bir ortam (`virtual environment`) oluşturun ve etkinleştirin (önerilir):**
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

4.  **Google Gemini API Anahtarınızı ayarlayın:**
    Uygulama, yapay zeka notları oluşturmak için bir `Google Gemini API Key` gerektirir.
    *   API anahtarınızı [Google AI Studio](https://aistudio.google.com/app/apikey) adresinden edinin.
    *   Projenin kök dizininde `.env` adında bir dosya oluşturun (ör. `Zettelkasten-AI-Notes/.env`).
    *   API anahtarınızı bu dosyaya aşağıdaki formatta ekleyin:
        ```
        GEMINI_API_KEY="API_ANAHTARINIZI_BURAYA_GIRIN"
        ```
    *   Alternatif olarak, API anahtarını doğrudan uygulama içinden `Settings -> Enter Gemini API Key` menü seçeneği aracılığıyla girebilirsiniz.

## Usage (Kullanım)

1.  **Uygulamayı başlatın:**
    ```bash
    python src/main.py
    ```

2.  **Temel Not Yönetimi:**
    *   **New Note:** Düzenleyiciyi temizlemek ve yeni bir nota başlamak için "New Note" düğmesine tıklayın.
    *   **Save Note:** `Markdown` içeriğinizi düzenleyiciye yazın. İlk satır otomatik olarak notun başlığı olacaktır. Veritabanına kaydetmek için "Save Note" düğmesine tıklayın.
    *   **Rename Note:** Listeden bir not seçin, ardından başlığını değiştirmek için "Rename Note" düğmesine tıklayın.
    *   **Delete Note:** Bir not seçin ve kaldırmak için "Delete Note" düğmesine tıklayın.

3.  **Kategori Yönetimi:**
    *   **Create New Category:** "New Category"ye tıklayın, bir ad girin ve o kategori içinde yeni bir not oluşturulacaktır.
    *   **Filter by Category:** Belirli bir kategoriye veya "All Notes"a ait notları görüntülemek için kategori açılır menüsünü kullanın.
    *   **Delete Category:** Açılır menüden bir kategori seçin ve "Delete Category" düğmesine tıklayın. Bu, kategoriyi ve onunla ilişkili tüm notları silecektir.

4.  **Markdown Preview:**
    Sol düzenleyici bölmesinde yazdıkça, sağ bölme notunuzun canlı bir `Markdown` önizlemesini gösterecektir.

5.  **AI Note Generation from PDF:**
    *   "Generate AI Notes from PDF" düğmesine tıklayın.
    *   Sisteminizden bir PDF dosyası seçin.
    *   Uygulama, PDF'ten metni çıkaracak ve not oluşturma için `Gemini AI`'ye gönderecektir.
    *   Oluşturulan notlar otomatik olarak kaydedilecek ve not listenize "AI Generated" kategorisi altında veya yapay zeka tarafından sağlanan genel bir başlık altında eklenecektir.

6.  **Note Linking:**
    *   **Link Notes:** Ana listeden bir not seçin. "Link Note" düğmesine tıklayın veya nota sağ tıklayıp "Link to..." seçeneğini seçin. Başka bir notu aramanıza ve seçmenize olanak tanıyan bir iletişim kutusu görünecektir.
    *   **View Linked Notes:** Ana listede bir not seçildiğinde, bağlantılı notları "Linked Notes List" bölmesinde görünecektir.
    *   **Open Linked Note:** Düzenleyicide açmak için "Linked Notes List"ndeki bağlantılı bir nota tıklayın.
    *   **Unlink Note:** Bağlantıyı kaldırmak için "Linked Notes List"ndeki bir nota sağ tıklayın ve "Unlink Note" seçeneğini seçin.

7.  **Mind Map Visualization:**
    *   Pencerenin altındaki zihin haritası, tüm notlarınızı `nodes` (düğümler) olarak ve bağlantılarını `connections` (bağlantılar) olarak görüntüler.
    *   İlgili notu düzenleyicide açmak için zihin haritasındaki bir `node`'a tıklayın.
    *   Görünümü `zoom in` ve `zoom out` yapmak için fare tekerleğini kullanın ve `pan` yapmak için sağ tıklayıp sürükleyin.

8.  **Theming:**
    *   Uygulamanın görünümünü değiştirmek için "Theme" menüsüne gidin ve "Light Theme" veya "Dark Theme"ı seçin.

## Project Structure (Proje Yapısı)
```
Zettelkasten-AI-Notes/
├── src/
│   ├── ai_note_generator_worker.py # AI notu oluşturmayı ayrı bir thread'de yönetir
│   ├── database_manager.py         # Tüm SQLite veritabanı işlemlerini yönetir
│   ├── gemini_api_client.py        # Google Gemini API ile arayüz oluşturur
│   ├── main.py                     # Ana uygulama giriş noktası ve GUI mantığı
│   ├── note_manager.py             # Notla ilgili işlemleri yönetir (save, rename, delete, sanitize)
│   ├── pdf_processor.py            # PDF dosyalarından metin içeriğini çıkarır
│   ├── mind_map_widget.py          # Zihin haritası görselleştirme widget'ı
│   ├── dark_theme.qss              # Karanlık tema için stil sayfası
│   ├── light_theme.qss             # Aydınlık tema için stil sayfası
│   └── logger.py                   # Hata ayıklama için basit bir logger
├── .gitignore                      # Git yoksayma dosyası
├── LICENSE                         # Proje lisansı
├── README.md                       # Projeye genel bakış ve kurulum talimatları
└── requirements.txt                # Python bağımlılıkları
```

## Contributing (Katkıda Bulunma)
:
1.  Depoyu `fork`'layın.
2.  Yeni bir `branch` oluşturun (`git checkout -b feature/NewFeature`).
3.  Değişikliklerinizi yapın.
4.  Değişikliklerinizi `commit`'leyin (`git commit -m 'Yeni bir özellik ekle'`).
5.  `Branch`'e `push`'layın (`git push origin feature/NewFeature`).
6.  Bir `Pull Request` açın.

## License (Lisans)
Bu proje MIT Lisansı altında lisanslanmıştır - ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

## Contact (İletişim)
Herhangi bir soru veya geri bildirim için lütfen GitHub deposunda bir `issue` açın.
