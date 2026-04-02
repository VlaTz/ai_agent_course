import logging
import os
from datetime import datetime
from pathlib import Path


class LoggerSetup:
    """Синглтон для настройки логирования"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._setup_root_logger()

    def _setup_root_logger(self):
        """Настройка корневого логгера"""

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Имя файла с датой
        log_filename = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Настраиваем корневой логгер
        root_logger = logging.getLogger()

        if root_logger.handlers:
            root_logger.handlers.clear()

        root_logger.setLevel(logging.INFO)

        # Файловый обработчик
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Получить логгер с указанным именем"""
        return logging.getLogger(name)


# Создаем единственный экземпляр
logger_setup = LoggerSetup()


def get_logger(name: str) -> logging.Logger:
    """Удобная функция для получения логгера"""
    return logger_setup.get_logger(name)