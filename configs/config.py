import os

class AppConfig:
    def __init__(self):
        self.port = os.getenv("APP_PORT") or '8080'