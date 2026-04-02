import asyncio
import json
from config_manager import settings
from service_functions import (_buttons_from_actions, _decode_button_id, _action_stop_item, _action_call_courier,
                               _action_call_chief)
from agent.one_agent import Agent
import os
from typing import Any
from fastapi import FastAPI, HTTPException
from logger_config import get_logger
from langchain_core.messages import HumanMessage


logger = get_logger('app')
agent = Agent(settings.agent)

# Полное время выполнения анализа (оба субагента + агрегатор); при превышении — пустой ответ.
ANALYZE_TIMEOUT_SEC = 90.0

app = FastAPI(title="LLM Analysis Service", version="1.0")

PORT = int(os.environ.get("LLM_SERVICE_PORT", "8001"))
HOST = os.environ.get("LLM_SERVICE_HOST", "0.0.0.0")

ACTION_HANDLERS = {
    "stop_item": _action_stop_item,
    "call_courier": _action_call_courier,
    "call_chief": _action_call_chief,
}


@app.post("/analyze")
async def analyze(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Принимает JSON с датчиками, возвращает {"response": "текст", "buttons": [{"id": "...", "label": "..."}, ...]}."""
    metrics = payload or {}
    try:
        result, actions, called_tools = await asyncio.wait_for(agent.run(metrics), timeout=ANALYZE_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning("Анализ превысил %s с — возвращаем пустой ответ", ANALYZE_TIMEOUT_SEC)
        return {"response": "", "buttons": []}
    final_answer = result['messages'][-1].content

    logger.info(f"Финальный ответ: {final_answer}")
    return {
        "response": final_answer,
        "buttons": _buttons_from_actions(actions),
    }

@app.post("/collect_metrics")
async def analyze(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Принимает JSON с датчиками, возвращает {"response": "текст", "buttons": [{"id": "...", "label": "..."}, ...]}."""
    metric = payload or {}
    actions = [[], []]
    with open('metrics.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    metrics['data'].append(metric)
    with open('metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {
        "response": '',
        "buttons": _buttons_from_actions(actions),
    }


@app.post("/action")
async def action(body: dict[str, Any] | None = None) -> dict[str, str]:
    """
    Выполняет действие по нажатой кнопке.
    Тело: {"button_id": "<id из ответа /analyze>"}. Возвращает {"result": "текст"}.
    """
    try:
        data = body or {}
        button_id = data.get("button_id") or ""
        if not button_id:
            raise HTTPException(status_code=400, detail="Не указан button_id")
        try:
            payload = _decode_button_id(button_id)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Некорректный button_id: {button_id}")
        action_type = payload.get("type")
        action_data = payload.get("data") or {}
        agent_key = payload.get("agent_key")
        if action_type not in ACTION_HANDLERS:
            raise HTTPException(status_code=400, detail=f"Неизвестное действие: {action_type}")
        fn = ACTION_HANDLERS[action_type]

        result = fn(action_data)
        if agent_key == "items":
            agent.item_agent.messages['messages'].append(HumanMessage(content=result))
        elif agent_key == "resources":
            agent.resources_agent.messages['messages'].append(HumanMessage(content=result))
        else:
            # Фолбэк для старых кнопок без agent_key.
            agent.messages['messages'].append(HumanMessage(content=result))
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
