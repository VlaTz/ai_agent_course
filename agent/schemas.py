from typing import Optional, Literal, Union
from pydantic import BaseModel


class CallChef(BaseModel):
    """Вызов повара"""
    employee_name: Optional[str] = None  # Имя повара, если конкретный
    time: Optional[str] = None  # Время, к которому нужен


class CallCourier(BaseModel):
    """Вызов курьера"""
    employee_name: Optional[str] = None  # Имя курьера, если конкретный
    time: Optional[str] = None  # Время, к которому нужен


class StopItem(BaseModel):
    """Поставить сырье в стоп"""
    item_name: Literal["тесто", "сыр", "соус", "пепперони", "грибы"] = ""  # Название сырья


class Action(BaseModel):
    """Общая структура действия"""
    priority: Literal["critical", "high", "medium", "low"]
    data: Union[CallChef, CallCourier, StopItem]   # Данные действия (одна из моделей выше)

