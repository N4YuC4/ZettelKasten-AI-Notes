import datetime
import os

# Hata ayiklama mesajlarini konsola ve bir log dosyasiyaz an yardimci fonksiyon
def log_debug(msg):
    print(msg) # Mesaji konsola yazdir
    
    # Log dizininin var oldugundan emin ol
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # logs/debug.log dosyasiina zaman damgasi ile birlikte mesaji ekle
    with open(os.path.join(log_dir, "debug.log"), "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] {msg}\n")