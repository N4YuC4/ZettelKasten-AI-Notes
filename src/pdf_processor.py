# pdf_processor.py
#
# Bu dosya, PDF dosyalarından metin çıkarmak için yardımcı işlevleri içerir.
# PyPDF2 kütüphanesini kullanarak bir PDF dosyasının tüm sayfalarındaki metni çıkarır.

import PyPDF2 # PDF dosyalarını okumak ve işlemek için kütüphane

# extract_text_from_pdf fonksiyonu, belirli bir PDF dosyasından metin çıkarır.
# pdf_path (str): PDF dosyasının yolu.
# Dönüş: PDF'ten çıkarılan metin (str) veya bir hata oluşursa None.
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    Args:
        pdf_path (str): The path to the PDF file.
    Returns:
        str: The extracted text from the PDF.
    """
    text = "" # Çıkarılan metni saklamak için boş bir string
    try:
        with open(pdf_path, "rb") as file: # PDF dosyasını ikili modda oku
            reader = PyPDF2.PdfReader(file) # PdfReader nesnesi oluştur
            for page_num in range(len(reader.pages)): # Her sayfa için döngü
                text += reader.pages[page_num].extract_text() # Sayfadaki metni çıkar ve ekle
    except Exception as e:
        print(f"Error extracting text from PDF: {e}") # Hata oluşursa yazdır
        return None # None döndür
    return text # Çıkarılan tüm metni döndür

# Bu blok, dosya doğrudan çalıştırıldığında örnek kullanım sağlar (test amaçlı).
if __name__ == '__main__':
    # Örnek kullanım (test amaçlı)
    # Eğer bir PDF dosyanız yoksa test için sahte bir PDF dosyası oluşturabilirsiniz.
    # from reportlab.pdfgen import canvas
    # c = canvas.Canvas("dummy.pdf")
    # c.drawString(100, 750, "This is a test PDF document.")
    # c.save()

    # extracted_text = extract_text_from_pdf("dummy.pdf")
    # if extracted_text:
    #     print("Extracted Text:")
    #     print(extracted_text)
    pass
