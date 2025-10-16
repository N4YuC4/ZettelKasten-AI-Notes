import datetime
import os

# Helper function that writes debug messages to the console and a log file
def log_debug(msg):
    print(msg) # Print the message to the console
    
    # Ensure the log directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Append the message with a timestamp to the logs/debug.log file
    with open(os.path.join(log_dir, "debug.log"), "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] {msg}\n")