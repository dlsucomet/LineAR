import sys
from datetime import datetime
import config

def init_session_files():
    with open(config.LOG_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(f"=== LineAR System Session Log Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

def log_message(message):
    config.status_msg = message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry, end="")
    try:
        with open(config.LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed writing to file log: {str(e)}", file=sys.stderr)