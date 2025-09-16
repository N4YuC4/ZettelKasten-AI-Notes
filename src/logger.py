import datetime

# Hata ayıklama mesajlarını konsola ve bir log dosyasına yazan yardımcı fonksiyon
def log_debug(msg):
    print(msg) # Mesajı konsola yazdır
    # logs/debug.log dosyasına zaman damgası ile birlikte mesajı ekle
    with open("logs/debug.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] {msg}\n")

