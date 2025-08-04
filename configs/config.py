import os

class AppConfig:
    def __init__(self):
        self.port = os.getenv("APP_PORT") or '8080'
        self.threshold_answer = os.getenv("THRESHOLD_ANSWER") or 0
        self.persistent_path = os.getenv("PERSISTENT_PATH") or './persistent'
        self.row = os.getenv("ROW") or 55
        self.col = os.getenv("COLUMN") or 40
        self.host = os.getenv("HOST") or "http://localhost:8080"
        self.min_accuracy = os.getenv("MIN_ACCURACY") or 0.75