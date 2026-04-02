import json
import base64
from agent.schemas import CallChef, CallCourier, StopItem
from typing import Any


def _encode_button_id(action_type: str, data: dict, agent_key: str | None = None) -> str:
    payload_dict = {"type": action_type, "data": data}
    if agent_key:
        payload_dict["agent_key"] = agent_key
    payload = json.dumps(payload_dict, ensure_ascii=False)
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_button_id(button_id: str) -> dict:
    pad = "=" * (-len(button_id) % 4)
    raw = base64.urlsafe_b64decode(button_id + pad)
    return json.loads(raw.decode("utf-8"))


def _action_stop_item(data: dict) -> str:
    item_name = data.get("item_name") or "?"
    return f"Ингредиент «{item_name}» поставлен в стоп"


def _action_call_courier(data: dict) -> str:
    name = data.get("employee_name") or "курьера"
    time = data.get("time")
    extra = f" к {time}" if time else ""
    return f"Прошу курьера «{name}» выйти пораньше{extra}"


def _action_call_chief(data: dict) -> str:
    name = data.get("employee_name") or "повара"
    time = data.get("time")
    extra = f" к {time}" if time else ""
    return f"Прошу повара «{name}» выйти пораньше{extra}"


def _flatten_actions(raw_actions: Any) -> list[tuple[str | None, Any]]:
    """
    Приводит actions к плоскому списку пар (agent_key, action):
    - старый формат: list[Action]
    - текущий формат менеджера: {"items": [[...]], "resources": [[...]]}
    """
    out: list[tuple[str | None, Any]] = []
    if raw_actions is None:
        return out

    def _append(agent_key: str | None, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for x in value:
                _append(agent_key, x)
            return
        out.append((agent_key, value))

    if isinstance(raw_actions, dict):
        for key, value in raw_actions.items():
            _append(str(key), value)
        return out

    _append(None, raw_actions)
    return out


def _buttons_from_actions(actions: list) -> list[dict[str, str]]:
    buttons_out: list[dict[str, str]] = []

    for agent_key, act in _flatten_actions(actions):
        action_type: str | None = None
        data_dict: dict[str, Any] = {}
        label: str | None = None

        # Старый формат: Action(priority, data, button_text, description, type, ...)
        if hasattr(act, "type") and hasattr(act, "data"):
            action_type = getattr(act, "type", None)
            data_obj = getattr(act, "data", None)
            data_dict = data_obj.model_dump() if hasattr(data_obj, "model_dump") else {}
            label = (getattr(act, "button_text", None) or getattr(act, "description", None) or action_type) or action_type

        elif isinstance(act, StopItem):
            action_type = "stop_item"
            data_dict = {"item_name": act.item_name}
            label = f"Поставить {act.item_name} в стоп"

        elif isinstance(act, CallCourier):
            action_type = "call_courier"
            data_dict = {
                "employee_name": act.employee_name,
                "time": act.time,
            }
            label = f"Вызвать курьера {act.employee_name} к {act.time}".strip()

        elif isinstance(act, CallChef):
            action_type = "call_chief"
            data_dict = {
                "employee_name": act.employee_name,
                "time": act.time,
            }
            label = f"Вызвать повара {act.employee_name} к {act.time}".strip()


        if not action_type:
            continue

        bid = _encode_button_id(action_type, data_dict, agent_key=agent_key)
        buttons_out.append({"id": bid, "label": (label or action_type).strip()})

    return buttons_out
