""" Функции для получения данных из конфигурационного файла. """
import inspect
import os
from typing import Dict

from dynaconf import Dynaconf

_file_name = inspect.getfile(inspect.currentframe())

config = os.path.join(os.path.dirname(__file__), 'config.json')
settings = Dynaconf(
    settings_files=[config],
    environments=True,
    auto_cast=True,
    envvar_prefix='DSA'
)

