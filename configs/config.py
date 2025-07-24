import os

class AppConfig:
    def __init__(self):
        self.port = os.getenv("APP_PORT") or '8080'
        self.threshold_answer = os.getenv("THRESHOLD_ANSWER") or 0
        self.persistent_path = os.getenv("PERSISTENT_PATH") or './persistent'
        self.row = os.getenv("ROW") or 4
        self.col = os.getenv("COLUMN") or 4
        self.host = os.getenv("COLUMN") or "http://localhost:5000"