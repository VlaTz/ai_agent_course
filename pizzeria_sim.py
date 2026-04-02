import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
import requests

from db import (
    init_db,
    get_orders_df,
    import_orders_from_excel,
    get_ingredients,
    save_ingredients,
    DEFAULT_INGREDIENTS,
    DEFAULT_INGREDIENT_UNITS,
    DEFAULT_AMOUNT_PER_PIZZA,
)

# ========== НАСТРОЙКИ СТРАНИЦЫ ==========
st.set_page_config(
    page_title="Пиццерия Симулятор на реальных данных",
    page_icon="🍕",
    layout="wide"
)


# ========== УТИЛИТЫ ДЛЯ ВРЕМЕНИ ==========
def parse_time_to_minutes(val):
    """Парсит время из строки 'H:MM:SS' или timedelta в минуты."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s == '0:00:00' or s == '00:00:00':
        return 0.0
    parts = s.replace(',', '.').split(':')
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 60 + m + sec / 60
    return 0.0


def send_sensors_to_external_system(payload: dict) -> bool:
    """
    Отправляет снимок датчиков и метрик во внешнюю систему (POST JSON).
    """

    if "last_external_payload" not in st.session_state:
        st.session_state.last_external_payload = None
    st.session_state.last_external_payload = payload
    return True


def _round_metric(v, ndigits=2):
    """Округляет числовую метрику до ndigits знаков после запятой; None остаётся 0."""
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return round(float(v), ndigits)
    return v


def build_external_payload(current_time, metrics: dict, sensors: dict, ingredient_units: dict, amount_per_pizza: dict = None) -> dict:
    """Формирует JSON для отправки: время снимка, показатели мониторинга,
    ингредиенты (количество, единица, расход на 1 пиццу). Числа — до 2 знаков после запятой.
    """
    amount_per_pizza = amount_per_pizza or {}
    names = {
        'dough': 'тесто',
        'cheese': 'сыр',
        'sauce': 'соус',
        'pepperoni': 'пепперони',
        'mushrooms': 'грибы'
    }

    ingredients = [
        {
            "ingredient": names[name],
            #"quantity": round(float(value), 2) if isinstance(value, (int, float)) else value,
            #"unit": ingredient_units.get(name, "шт"),
            "pizzas_rate": round(float(value) / float(amount_per_pizza.get(name, 0)), 2),
        }
        for name, value in sensors.items()
    ]
    return {
        "timestamp": current_time.isoformat() if hasattr(current_time, "isoformat") else str(current_time),
        "metrics": {
            "orders_in_work_hall": metrics.get("orders_in_work_hall", 0),
            "orders_in_work_delivery": metrics.get("orders_in_work_delivery", 0),
            "orders_per_chef": _round_metric(metrics.get("orders_per_chef", 0)),
            "orders_per_courier": _round_metric(metrics.get("orders_per_courier", 0)),
            "mean_courier_trip_min": _round_metric(metrics.get("mean_courier_trip_min")),
            "avg_cooking_hall_min": _round_metric(metrics.get("avg_cooking_hall_min")),
            "avg_cooking_delivery_min": _round_metric(metrics.get("avg_cooking_delivery_min")),
            "avg_shelf_min": _round_metric(metrics.get("avg_shelf_min")),
        },
        "ingredients": ingredients,
    }


# ========== АНАЛИЗ ДАННЫХ ЧЕРЕЗ LLM ==========
LLM_SERVICE_URL_DEFAULT = os.environ.get("LLM_SERVICE_URL", "http://localhost:8001/analyze")


def call_llm_analysis(payload: dict, service_url: str, timeout_s: float = 100.0) -> dict:
    """
    Отправляет снимок датчиков в LLM-сервис (POST /analyze).
    Возвращает {"response": "текст", "buttons": [{"id": "...", "label": "..."}, ...]}.
    """
    if not service_url:
        return {"response": "", "buttons": [], "ok": False}

    try:
        r = requests.post(service_url, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            return {"response": "", "buttons": [], "ok": False}
        data = r.json() if r.content else {}
        text = data.get("response", "")
        buttons = data.get("buttons") or []
        if not isinstance(buttons, list):
            buttons = []
        buttons = [b for b in buttons if isinstance(b, dict) and b.get("id") and b.get("label")]
        return {"response": text if isinstance(text, str) else "", "buttons": buttons, "ok": True}
    except Exception:
        return {"response": "", "buttons": [], "ok": False}


def call_llm_action(service_base_url: str, button_id: str, timeout_s: float = 10.0) -> str:
    """
    Вызывает действие на стороне LLM-сервиса (POST /action).
    Возвращает текст результата или пустую строку.
    """
    if not service_base_url or not button_id:
        return ""
    action_url = service_base_url.rstrip("/").replace("/analyze", "") + "/action"
    try:
        r = requests.post(action_url, json={"button_id": button_id}, timeout=timeout_s)
        if r.status_code != 200:
            return ""
        data = r.json() if r.content else {}
        return data.get("result", "") if isinstance(data.get("result"), str) else ""
    except Exception:
        return ""


# ========== ЗАГРУЗКА ДАННЫХ ИЗ БД ==========
def load_orders():
    """Загружает заказы из БД. Если таблица пуста — пробует импорт из data.xlsx."""
    init_db()
    df = get_orders_df()
    if df.empty:
        try:
            import_orders_from_excel("data.xlsx", parse_time_to_minutes)
            df = get_orders_df()
        except FileNotFoundError:
            pass
    return df


# ========== МОНИТОРИНГ ПОКАЗАТЕЛЕЙ ==========
def count_orders_in_current_sim_hour(df: pd.DataFrame, current_time: datetime) -> int:
    """
    Заказы, поступившие с начала текущего часа симуляции до current_time включительно
    (по полю order_time). Дубликаты order_id / order_time учитываются как один заказ.
    """
    hour_start = current_time.replace(minute=0, second=0, microsecond=0)
    mask = (df["order_time"] >= hour_start) & (df["order_time"] <= current_time)
    sub = df.loc[mask]
    if sub.empty:
        return 0
    if "order_id" in sub.columns:
        sub = sub.drop_duplicates(subset=["order_id"], keep="first")
    else:
        sub = sub.drop_duplicates(subset=["order_time"], keep="first")
    return int(len(sub))


def compute_metrics_at_time(df: pd.DataFrame, current_time: datetime, window_minutes: int) -> dict:
    """
    Заказы в работе — срез на current_time (старая логика).
    В окне [current_time - window_minutes, current_time]: среднее время приготовления (зал/доставка),
    среднее время поездки курьера и среднее время на тепловой полке.
    """
    # Заказы в работе на текущий момент (срез, без окна)
    started = df['order_time'] <= current_time
    not_finished = df['completion_time'] > current_time
    in_work = started & not_finished

    in_work_hall = (in_work & (df['order_type'] == 'rest')).sum()
    in_work_delivery = (in_work & (df['order_type'] == 'del')).sum()

    # Окно: только среднее время приготовления и поездки курьера
    window_start = current_time - timedelta(minutes=window_minutes)
    completed_in_window = (df['completion_time'] > window_start) & (df['completion_time'] <= current_time)
    completed_hall_window = df[completed_in_window & (df['order_type'] == 'rest') & (df['cooking_min'] != 0)]
    completed_del_window = df[completed_in_window & (df['order_type'] == 'del')]

    avg_cooking_hall = completed_hall_window['cooking_min'].mean() if len(completed_hall_window) > 0 else None
    avg_cooking_delivery = completed_del_window['cooking_min'].mean() if len(completed_del_window) > 0 else None
    mean_courier = (completed_del_window['courier_trip_min'].mean()
                    if len(completed_del_window) > 0 else None)

    # Среднее время на тепловой полке — в том же окне
    avg_shelf = df.loc[completed_in_window, 'shelf_min'].mean() if completed_in_window.any() else None

    # Заказы на повара = (заказы в работе) / (кол-во поваров на смене в текущем часе)
    def _normalize_hour_str(val) -> str:
        s = str(val).strip()
        # В графике могут встречаться артефакты вроде ",16:00"
        s = s.replace(",", "").strip()
        if not s:
            return ""
        try:
            return pd.to_datetime(s).strftime("%H:00")
        except Exception:
            return s

    chefs_on_shift = 0
    orders_in_work_total = int(in_work_hall) + int(in_work_delivery)
    current_hour_key = current_time.strftime("%H:00")
    try:
        with open("data/schedule.json", "r", encoding="utf-8") as f:
            schedule = json.load(f)
        for chef in schedule:
            for h in (chef.get("hours", []) or []):
                if _normalize_hour_str(h) == current_hour_key:
                    chefs_on_shift += 1
                    break
    except Exception:
        chefs_on_shift = 0

    couriers_on_shift = 0
    try:
        with open("data/c_schedule.json", "r", encoding="utf-8") as f:
            c_schedule = json.load(f)
        for courier in c_schedule:
            for h in (courier.get("hours", []) or []):
                if _normalize_hour_str(h) == current_hour_key:
                    couriers_on_shift += 1
                    break
    except Exception:
        chefs_on_shift = 0

    orders_per_chef = (orders_in_work_total / chefs_on_shift) if chefs_on_shift > 0 else 0
    orders_per_courier = (int(in_work_delivery) / couriers_on_shift) if couriers_on_shift > 0 else 0

    orders_current_hour = count_orders_in_current_sim_hour(df, current_time)

    return {
        'orders_in_work_hall': int(in_work_hall),
        'orders_in_work_delivery': int(in_work_delivery),
        'orders_current_hour': orders_current_hour,
        'orders_per_chef': round(float(orders_per_chef), 2),
        'orders_per_courier': round(float(orders_per_courier), 2),
        'mean_courier_trip_min': mean_courier,
        'avg_cooking_hall_min': avg_cooking_hall,
        'avg_cooking_delivery_min': avg_cooking_delivery,
        'avg_shelf_min': avg_shelf,
    }


def apply_chaos_overlay(metrics: dict, order_mult: float, time_mult: float) -> dict:
    """
    Копия метрик с искусственным «перегрузом» для теста агента.
    Исходный DataFrame заказов и файлы не используются и не меняются.
    """
    out = dict(metrics)
    out['orders_in_work_hall'] = max(0, int(round(out.get('orders_in_work_hall', 0) * order_mult)))
    out['orders_in_work_delivery'] = max(0, int(round(out.get('orders_in_work_delivery', 0) * order_mult)))
    out['orders_current_hour'] = max(0, int(round(out.get('orders_current_hour', 0) * order_mult)))
    # Знаменатель (повара) не меняется, поэтому заказы на повара масштабируются так же, как и заказы в работе.
    if 'orders_per_chef' in out and isinstance(out.get('orders_per_chef'), (int, float)):
        out['orders_per_chef'] = max(0.0, float(out.get('orders_per_chef', 0.0)) * order_mult)
    for key in ('avg_cooking_hall_min', 'avg_cooking_delivery_min', 'mean_courier_trip_min', 'avg_shelf_min'):
        v = out.get(key)
        if v is not None and isinstance(v, (int, float)):
            out[key] = float(v) * time_mult
    return out


def compute_metrics_effective(df: pd.DataFrame, current_time: datetime, window_minutes: int) -> dict:
    """Метрики из данных; при включённом режиме хаоса — только искажённая копия для UI и агента."""
    m = compute_metrics_at_time(df, current_time, window_minutes)
    if not st.session_state.get("chaos_enabled"):
        return m
    om = float(st.session_state.get("chaos_order_mult", 2.5))
    tm = float(st.session_state.get("chaos_time_mult", 1.8))
    return apply_chaos_overlay(m, om, tm)


# ========== СИМУЛЯТОР (время + ингредиенты из БД) ==========
class PizzeriaSimulator:
    """Единый запуск: время по заказам из БД, ингредиенты и расход на 1 пиццу из БД."""

    def __init__(self, orders_df, initial_sensors=None, initial_units=None, initial_amount_per_pizza=None):
        self.orders_df = orders_df
        self.current_time = orders_df['order_time'].min()
        end_completion = orders_df['completion_time'].max() if 'completion_time' in orders_df.columns else orders_df['order_time'].max()
        self.end_time = max(end_completion, orders_df['order_time'].max())
        self.sensors = dict(initial_sensors or DEFAULT_INGREDIENTS)
        self.ingredient_units = dict(
            initial_units or {k: DEFAULT_INGREDIENT_UNITS.get(k, "шт") for k in self.sensors}
        )
        self.amount_per_pizza = dict(
            initial_amount_per_pizza or {k: DEFAULT_AMOUNT_PER_PIZZA.get(k, 0) for k in self.sensors}
        )
        self.history = []

    def step(self, step_minutes):
        """Шаг: сдвиг времени на step_minutes и списание ингредиентов по заказам, поступившим в этом шаге. Каждый заказ учитывается не более одного раза."""
        new_time = self.current_time + timedelta(minutes=step_minutes)
        if new_time > self.end_time:
            new_time = self.end_time

        # Заказы, поступившие в текущем шаге (order_time в (current_time, new_time])
        arrived = self.orders_df[
            (self.orders_df['order_time'] > self.current_time) &
            (self.orders_df['order_time'] <= new_time)
        ]
        # Убираем дубликаты по заказу, чтобы не списать дважды за один заказ (даже если строка попала в окно несколько раз)
        if 'order_id' in arrived.columns:
            arrived = arrived.drop_duplicates(subset=['order_id'], keep='first')
        else:
            arrived = arrived.drop_duplicates(subset=['order_time'], keep='first')

        for _, order in arrived.iterrows():
            n = max(1, int(order.get('products_count', 1)))

            for name in list(self.sensors.keys()):
                amt = self.amount_per_pizza.get(name, 0)
                if amt <= 0:
                    continue
                need = n * amt
                if name == 'dough':
                    self.sensors['dough'] -= need
                else:
                    self.sensors[name] -= min(need, max(0, self.sensors[name]))
        for key in self.sensors:
            if isinstance(self.sensors[key], (int, float)):
                self.sensors[key] = max(0, round(float(self.sensors[key]), 2))

        self.current_time = new_time
        self.history.append({'time': self.current_time, **self.sensors})
        return self.sensors

    def is_finished(self):
        return self.current_time >= self.end_time

    def reset(self, orders_df):
        self.orders_df = orders_df
        self.current_time = orders_df['order_time'].min()
        end_completion = orders_df['completion_time'].max() if 'completion_time' in orders_df.columns else orders_df['order_time'].max()
        self.end_time = max(end_completion, orders_df['order_time'].max())
        data = get_ingredients()
        self.sensors = {x["name"]: x["value"] for x in data}
        self.ingredient_units = {x["name"]: x["unit"] for x in data}
        self.amount_per_pizza = {x["name"]: x["amount_per_pizza"] for x in data}
        self.history = []


# ========== UI (единый интерфейс) ==========

st.title("🍕 Пиццерия — мониторинг")
#st.caption("Заказы и ингредиенты хранятся в БД. При первом запуске заказы импортируются из data.xlsx.")

df_orders = load_orders()
if df_orders.empty:
    st.warning("В БД нет заказов. Положите data.xlsx в папку с приложением и обновите страницу — заказы импортируются автоматически.")
    st.stop()


def _parse_hour_to_float(h: str) -> float:
    """
    "HH:MM" -> часы в виде float.
    Нужен для визуализации расписаний на оси 0..24.
    """
    s = str(h).strip().replace(",", "").strip()
    if not s:
        return 0.0
    try:
        t = pd.to_datetime(s).time()
        return float(t.hour) + float(t.minute) / 60.0
    except Exception:
        return 0.0


def _load_schedule(path: str) -> list[dict]:
    """Читает список смен из json (list[{"name":..,"shift":{"start":..,"end":..}}])."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _schedule_figure(schedule: list[dict], title: str, color: str) -> go.Figure:
    """
    Горизонтальный график смен (аналог Gantt):
    по Y — сотрудник, по X — интервал [start, end] в часах (0..24).
    """
    rows = []
    for s in schedule or []:
        name = str(s.get("name", "")).strip() or "—"
        shift = s.get("shift") or {}
        start = _parse_hour_to_float(shift.get("start", "00:00"))
        end = _parse_hour_to_float(shift.get("end", "00:00"))
        # Смена до полуночи в данных часто как "00:00" — считаем как 24:00.
        if end == 0.0 and str(shift.get("end", "")).strip().startswith("00"):
            end = 24.0
        # На всякий случай (если start/end перепутаны)
        if end < start:
            end = start
        rows.append((name, start, end))

    # Сортировка: сначала ранние смены
    rows.sort(key=lambda x: (x[1], x[2], x[0]))

    fig = go.Figure()
    for (name, start, end) in rows:
        dur = max(0.0, float(end) - float(start))
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=[name],
                x=[dur],
                base=[start],
                marker_color=color,
                hovertemplate="Сотрудник: %{y}<br>Смена: %{base:.0f}:00–%{customdata:.0f}:00<extra></extra>",
                customdata=[end],
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        height=max(260, 28 * max(6, len(rows))),
        barmode="overlay",
        xaxis=dict(
            title="Час",
            range=[8, 24],
            tickmode="array",
            tickvals=list(range(0, 25, 1)),
            ticktext=[f"{h:02d}:00" for h in range(0, 25, 1)],
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(title=""),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

# Данные погоды (используем для простого мониторинга на дашборде).
try:
    df_weather = pd.read_excel("data/weather.xlsx")
except Exception:
    df_weather = None

with st.expander("📋 Заказы из БД"):
    rest_count = len(df_orders[df_orders['order_type'] == 'rest'])
    del_count = len(df_orders[df_orders['order_type'] == 'del'])
    total_products = int(df_orders['products_count'].sum())
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Всего заказов", len(df_orders))
    with col_b:
        st.metric("🍽️ В зал", rest_count)
    with col_c:
        st.metric("🛵 Доставка", del_count)
    with col_d:
        st.metric("🍕 Всего пицц (продуктов) за день", total_products)
    st.dataframe(df_orders.head(20))

# Единое состояние симуляции (ингредиенты из БД)
if 'sim' not in st.session_state:
    _ing = get_ingredients()
    _sensors = {x["name"]: x["value"] for x in _ing}
    _units = {x["name"]: x["unit"] for x in _ing}
    _amount = {x["name"]: x["amount_per_pizza"] for x in _ing}
    st.session_state.sim = PizzeriaSimulator(df_orders, _sensors, _units, _amount)
    st.session_state.metric_history = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'interval_min' not in st.session_state:
    st.session_state.interval_min = 15
if 'speed' not in st.session_state:
    st.session_state.speed = 1
if 'llm_chat_history' not in st.session_state:
    st.session_state.llm_chat_history = []
if "llm_enabled" not in st.session_state:
    st.session_state.llm_enabled = True
if "llm_service_url" not in st.session_state:
    st.session_state.llm_service_url = LLM_SERVICE_URL_DEFAULT
if "last_llm_response" not in st.session_state:
    st.session_state.last_llm_response = ""
if "last_llm_timestamp" not in st.session_state:
    st.session_state.last_llm_timestamp = ""
if "last_llm_buttons" not in st.session_state:
    st.session_state.last_llm_buttons = []
if "last_action_result" not in st.session_state:
    st.session_state.last_action_result = ""
if "waiting_for_llm" not in st.session_state:
    st.session_state.waiting_for_llm = False
if "pending_llm_payload" not in st.session_state:
    st.session_state.pending_llm_payload = None
if "chaos_enabled" not in st.session_state:
    st.session_state.chaos_enabled = False
if "chaos_order_mult" not in st.session_state:
    st.session_state.chaos_order_mult = 2.5
if "chaos_time_mult" not in st.session_state:
    st.session_state.chaos_time_mult = 1.8

with st.sidebar:
    st.subheader(f"Время симуляции {st.session_state.sim.current_time.strftime('%H:%M:%S')}")
    st.toggle("Включить LLM-анализ", key="llm_enabled")
    if st.session_state.last_llm_timestamp:
        st.caption(f"Последний ответ: {st.session_state.last_llm_timestamp}")
    st.markdown("### Последний ответ LLM")
    if st.session_state.last_llm_response:
        # Markdown рендерится через st.markdown, в отличие от st.text_area.
        st.markdown(st.session_state.last_llm_response)
    else:
        st.info("Ответ пока пуст.")
    # Кнопки от LLM (обработка на стороне сервиса)
    service_base = (st.session_state.llm_service_url or "").rstrip("/").replace("/analyze", "").strip() or "http://localhost:8001"
    for btn in st.session_state.last_llm_buttons:
        if st.button(btn.get("label", btn.get("id", "")), key=f"llm_btn_{btn.get('id', '')}"):
            res = call_llm_action(service_base, btn.get("id", ""))
            st.session_state.last_action_result = res or "Действие выполнено."
            st.rerun()
    if st.session_state.last_action_result:
        st.success(st.session_state.last_action_result)
    with st.expander("История ответов", expanded=False):
        if st.session_state.llm_chat_history:
            for idx, item in enumerate(reversed(st.session_state.llm_chat_history[-50:])):
                st.markdown(f"**{item['time']}**\n\n{item['response']}")
                for b in item.get("buttons") or []:
                    if st.button(b.get("label", b.get("id")), key=f"hist_{idx}_{b.get('id')}"):
                        res = call_llm_action(service_base, b.get("id", ""))
                        st.session_state.last_action_result = res or "Действие выполнено."
                        st.rerun()
                st.divider()
        else:
            st.info("История пустая.")
    if st.button("🧹 Очистить историю LLM"):
        st.session_state.llm_chat_history = []
        st.session_state.last_llm_response = ""
        st.session_state.last_llm_timestamp = ""
        st.session_state.last_llm_buttons = []
        st.session_state.last_action_result = ""
        st.session_state.waiting_for_llm = False
        st.session_state.pending_llm_payload = None
        st.rerun()

sim = st.session_state.sim
end_time = sim.end_time

# Вкладки UI
tab_monitoring, tab_graphs = st.tabs(["Мониторинг", "Графики"])

with tab_monitoring:
    # Панель управления (одна)
    st.subheader("⏱ Управление симуляцией")
    interval_min = st.number_input(
        "Интервал мониторинга (минут)",
        min_value=1,
        max_value=60,
        value=st.session_state.interval_min,
        step=1,
        key="interval_input"
    )
    st.session_state.interval_min = interval_min

    speed = st.select_slider(
        "Скорость симуляции",
        options=[0.5, 1, 2, 5, 10],
        value=st.session_state.speed,
        format_func=lambda x: f"{x}×",
        key="speed_slider"
    )
    st.session_state.speed = speed

    with st.expander("🧪 Режим хаоса (тест агента)", expanded=False):
        st.caption(
            "Временно усиливает заказы в работе и времена готовки/курьера/полки **только в метриках** "
            "для монитора и LLM. Заказы в БД и Excel не перезаписываются."
        )
        st.toggle("Включить искажение метрик", key="chaos_enabled")
        cx, cy = st.columns(2)
        with cx:
            st.slider(
                "Множитель заказов в работе",
                min_value=1.2,
                max_value=5.0,
                step=0.1,
                key="chaos_order_mult",
            )
        with cy:
            st.slider(
                "Множитель времён (готовка, курьер, полка)",
                min_value=1.1,
                max_value=3.0,
                step=0.1,
                key="chaos_time_mult",
            )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("▶️ Пуск"):
            st.session_state.running = True
            st.rerun()
    with c2:
        if st.button("⏸️ Пауза"):
            st.session_state.running = False
            st.rerun()
    with c3:
        if st.button("⏭️ Шаг"):
            if not sim.is_finished():

                m = compute_metrics_effective(df_orders, sim.current_time, interval_min)
                m['time'] = sim.current_time
                st.session_state.metric_history.append(m)
                payload = build_external_payload(sim.current_time, m, sim.sensors, sim.ingredient_units, sim.amount_per_pizza)
                send_sensors_to_external_system(payload)
                if st.session_state.llm_enabled:
                    llm_data = call_llm_analysis(payload, st.session_state.llm_service_url)
                    ts = payload.get("timestamp", "")[:19].replace("T", " ")
                    resp_text = llm_data.get("response", "").strip()
                    resp_buttons = llm_data.get("buttons", [])
                    if resp_text:
                        st.session_state.llm_chat_history.append({"time": ts, "response": resp_text, "buttons": resp_buttons})
                        st.session_state.last_llm_response = resp_text
                        st.session_state.last_llm_timestamp = ts
                        st.session_state.last_llm_buttons = resp_buttons
                        st.session_state.running = False  # пауза при непустом ответе LLM
            sim.step(interval_min)
            st.rerun()
    with c4:
        st.metric("Время симуляции", sim.current_time.strftime('%H:%M:%S'))
    with c5:
        if st.button("🔄 Сброс"):
            sim.reset(df_orders)
            st.session_state.metric_history = []
            st.session_state.llm_chat_history = []
            st.session_state.last_llm_response = ""
            st.session_state.last_llm_timestamp = ""
            st.session_state.last_llm_buttons = []
            st.session_state.last_action_result = ""
            st.session_state.waiting_for_llm = False
            st.session_state.pending_llm_payload = None
            st.session_state.running = False
            st.rerun()

    st.caption(f"Конец данных: {end_time.strftime('%Y-%m-%d %H:%M')}")

    # Заметный индикатор: симуляция идёт / пауза / ожидание LLM
    if st.session_state.running:
        if st.session_state.get("waiting_for_llm"):
            status_emoji, status_text = "⏳", "ОЖИДАНИЕ ОТВЕТА LLM"
            status_bg, status_color = "#fff3cd", "#856404"  # жёлтый
        else:
            status_emoji, status_text = "🔄", "СИМУЛЯЦИЯ ИДЁТ"
            status_bg, status_color = "#cce5ff", "#004085"  # синий
        status_state = "running"
    else:
        status_emoji, status_text = "⏸️", "ПАУЗА"
        status_bg, status_color = "#e2e3e5", "#383d41"  # серый
        status_state = "complete"

    st.markdown(
        f'<div style="background: {status_bg}; color: {status_color}; padding: 14px 20px; '
        f'border-radius: 8px; border-left: 6px solid {status_color}; margin: 12px 0; '
        f'font-size: 1.35rem; font-weight: 700;">{status_emoji} {status_text}</div>',
        unsafe_allow_html=True,
    )
    with st.status(status_text.lower(), state=status_state, expanded=True):
        if status_state == "running":
            st.caption("Выполняется шаг или запрос к LLM. Дождитесь ответа.")
        else:
            st.caption("Нажмите «Пуск» для продолжения.")

    # Заглушка: последний payload, который «отправился» во внешнюю систему
    with st.expander("📤 Отправка данных агенту"):
        st.caption("При каждом обновлении датчиков (каждые N минут) формируется payload и вызывается отправка. Пока реальный POST не выполняется — данные только сохраняются ниже.")
        if st.session_state.get("last_external_payload"):
            st.json(st.session_state.last_external_payload)
        else:
            st.info("Выполните «Шаг» или «Пуск» — здесь появится последний отправленный снимок.")

    # Калибровка ингредиентов: можно менять значение в UI, сохраняется в БД
    st.subheader("🥗 Калибровка ингредиентов")
    st.caption("Измените значение и нажмите «Применить» — данные сохранятся в БД и будут использоваться в симуляции.")
    INGREDIENT_LABELS = {
        "dough": "🍞 Тесто",
        "cheese": "🧀 Сыр",
        "sauce": "🥫 Соус",
        "pepperoni": "🥩 Пепперони",
        "mushrooms": "🍄 Грибы",
    }
    cal_cols = st.columns(5)
    new_sensors = {}
    for idx, name in enumerate(DEFAULT_INGREDIENTS):
        with cal_cols[idx]:
            unit = sim.ingredient_units.get(name, "шт")
            val = st.number_input(
                f"{INGREDIENT_LABELS[name]}, {unit}",
                min_value=0,
                value=int(sim.sensors.get(name, DEFAULT_INGREDIENTS[name])),
                step=1,
                key=f"cal_{name}",
            )
            new_sensors[name] = max(0, val)
    if st.button("💾 Применить и сохранить в БД"):
        for name, value in new_sensors.items():
            sim.sensors[name] = value
        save_ingredients(new_sensors)
        st.success("Ингредиенты сохранены в БД.")
        st.rerun()

    # Все датчики в одном блоке
    st.subheader("📊 Показатели мониторинга и датчики")
    if st.session_state.get("chaos_enabled"):
        st.warning(
            "⚠️ **Режим хаоса включён** — числа ниже и в payload для агента усилены; исходные данные заказов не меняются."
        )

    metrics = compute_metrics_effective(df_orders, sim.current_time, interval_min)
    s = sim.sensors

    # Строка 1: показатели из data.xlsx
    row1_1, row1_2, row1_3, row1_4, row1_5, row1_6, row1_7 = st.columns(7)
    with row1_1:
        st.metric("Заказов в работе (зал)", metrics['orders_in_work_hall'])
    with row1_2:
        st.metric("Заказов в работе (доставка)", metrics['orders_in_work_delivery'])
    with row1_3:
        st.metric("Заказов за текущий час", metrics.get('orders_current_hour', 0))
    with row1_4:
        v = metrics['mean_courier_trip_min']
        st.metric("Ср. время поездки, мин", f"{v:.1f}" if v is not None else "—")
    with row1_5:
        h = metrics['avg_cooking_hall_min']
        st.metric("Ср. время готовки (зал), мин", f"{h:.1f}" if h is not None else "—")
    with row1_6:
        d = metrics['avg_cooking_delivery_min']
        st.metric("Ср. время готовки (доставка), мин", f"{d:.1f}" if d is not None else "—")
    with row1_7:
        v = metrics['avg_shelf_min']
        st.metric("Ср. время на полке, мин", f"{v:.1f}" if v is not None else "—")

    # Строка 2: ингредиенты
    st.caption("Остатки ингредиентов (из БД, тратятся по заказам)")
    row2_1, row2_2, row2_3, row2_4, row2_5 = st.columns(5)
    with row2_1:
        u = sim.ingredient_units.get("dough", "шт")
        st.metric("🍞 Тесто", f"{s['dough']} {u}")
    with row2_2:
        u = sim.ingredient_units.get("cheese", "шт")
        st.metric("🧀 Сыр", f"{s['cheese']} {u}")
    with row2_3:
        u = sim.ingredient_units.get("sauce", "шт")
        st.metric("🥫 Соус", f"{s['sauce']} {u}")
    with row2_4:
        u = sim.ingredient_units.get("pepperoni", "шт")
        st.metric("🥩 Пепперони", f"{s['pepperoni']} {u}")
    with row2_5:
        u = sim.ingredient_units.get("mushrooms", "шт")
        st.metric("🍄 Грибы", f"{s['mushrooms']} {u}")

    # Два столбчатых графика (горизонтальные — столбцы сверху вниз)
    st.subheader("📊 Датчики")
    col_sensors, col_ing = st.columns(2)

    with col_sensors:
        labels_sensors = [
            "В работе (зал)", "В работе (доставка)", "Заказов за час",
            "Заказов на повара", "Медиана курьера, мин", "Ср. готовка зал", "Ср. готовка доставка", "Ср. полка, мин"
        ]
        values_sensors = [
            metrics['orders_in_work_hall'],
            metrics['orders_in_work_delivery'],
            metrics.get('orders_current_hour', 0),
            metrics.get('orders_per_chef', 0),
            metrics['mean_courier_trip_min'] or 0,
            metrics['avg_cooking_hall_min'] or 0,
            metrics['avg_cooking_delivery_min'] or 0,
            metrics['avg_shelf_min'] or 0,
        ]
        fig_sensors = go.Figure(data=[
            go.Bar(
                orientation="h",
                y=labels_sensors,
                x=values_sensors,
                marker_color="#4ECDC4",
                text=values_sensors,
                textposition="auto",
                texttemplate="%{x}",
            )
        ])
        fig_sensors.update_layout(
            title="Основные датчики",
            height=380,
            xaxis_title="Значение",
            yaxis_title="",
            yaxis_autorange="reversed",
            margin=dict(l=10),
            showlegend=False,
        )
        st.plotly_chart(fig_sensors, width='stretch')

        # Погода на текущий час симуляции (просто мониторинг).
        if df_weather is not None and not df_weather.empty and hasattr(sim, "current_time"):
            hour_col = df_weather.columns[0]
            desc_col = df_weather.columns[1] if len(df_weather.columns) > 1 else None
            cur_hour = int(sim.current_time.hour)
            subw = df_weather[df_weather[hour_col] == cur_hour]
            weather_text = "—"
            if desc_col is not None and len(subw) > 0:
                weather_text = str(subw.iloc[0][desc_col])
            st.metric(f"Погода на {cur_hour:02d}:00", weather_text)

    with col_ing:
        labels_ing = [f"Тесто, {sim.ingredient_units.get('dough', 'шт')}", f"Сыр, {sim.ingredient_units.get('cheese', 'шт')}", f"Соус, {sim.ingredient_units.get('sauce', 'шт')}", f"Пепперони, {sim.ingredient_units.get('pepperoni', 'шт')}", f"Грибы, {sim.ingredient_units.get('mushrooms', 'шт')}"]
        values_ing = [s['dough'], s['cheese'], s['sauce'], s['pepperoni'], s['mushrooms']]
        colors_ing = ["#F9D56E", "#FFE194", "#E3B23C", "#E57373", "#81C784"]
        fig_ing = go.Figure(data=[
            go.Bar(
                orientation="h",
                y=labels_ing,
                x=values_ing,
                marker_color=colors_ing,
                text=values_ing,
                textposition="auto",
                texttemplate="%{x}",
            )
        ])
        fig_ing.update_layout(
            title="Ингредиенты",
            height=320,
            xaxis_title="Значение",
            yaxis_title="",
            yaxis_autorange="reversed",
            margin=dict(l=10),
            showlegend=False,
        )
        st.plotly_chart(fig_ing, width='stretch')

with tab_graphs:
    st.subheader("👥 Графики смен сотрудников")

    chefs = _load_schedule("data/schedule.json")
    couriers = _load_schedule("data/c_schedule.json")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_schedule_figure(chefs, "Повара", "#FF6B6B"), width="stretch")
    with col_b:
        st.plotly_chart(_schedule_figure(couriers, "Курьеры", "#45B7D1"), width="stretch")

# Графики

# Для графиков сравнения с прошлой неделей используем benchmark из pre_data.xlsx
try:
    pre_data_bench = pd.read_excel('data/pre_data.xlsx')
    bench_hour_col = pre_data_bench.columns[0]
    bench_orders_col = pre_data_bench.columns[-1]
except Exception:
    pre_data_bench = None
    bench_hour_col = None
    bench_orders_col = None

# 1) Заказы в работе
st.subheader("📈 Заказы в работе")
if st.session_state.metric_history:
    hist = pd.DataFrame(st.session_state.metric_history)
    if 'orders_current_hour' not in hist.columns:
        hist['orders_current_hour'] = 0
    hist['time_str'] = hist['time'].apply(lambda t: t.strftime('%H:%M') if hasattr(t, "strftime") else str(t))
    fig_work = go.Figure()
    fig_work.add_trace(
        go.Scatter(
            x=hist['time_str'],
            y=hist['orders_in_work_hall'],
            mode='lines+markers',
            name='В работе (зал)',
            line=dict(color='#4ECDC4'),
        )
    )
    fig_work.add_trace(
        go.Scatter(
            x=hist['time_str'],
            y=hist['orders_in_work_delivery'],
            mode='lines+markers',
            name='В работе (доставка)',
            line=dict(color='#45B7D1'),
        )
    )
    fig_work.update_layout(
        title="Заказы в работе (зал и доставка)",
        height=280,
        xaxis_title="Время",
        yaxis_title="Кол-во",
    )
    st.plotly_chart(fig_work, width='stretch')
else:
    st.info("Нажмите «Шаг» или «Пуск» — появится график заказов в работе.")




# 2.1) Сравнение: заказы сегодня vs прошлой неделе по часам
st.subheader("📊 Заказы по часам: сегодня vs прошлой неделе")

current_hour = sim.current_time.hour  # граница для динамической линии "сегодня"

try:
    df_orders_dt = df_orders.copy()
    df_orders_dt["order_time"] = pd.to_datetime(df_orders_dt["order_time"])
    today_date = df_orders_dt["order_time"].dt.date.min()
except Exception:
    df_orders_dt = None
    today_date = None

def _count_orders_in_hour(date_val, hour: int, upper_time: datetime | None = None) -> int:
    if df_orders_dt is None or today_date is None:
        return 0
    hour_start = datetime(date_val.year, date_val.month, date_val.day, hour, 0, 0)
    hour_end = hour_start + timedelta(hours=1)
    if upper_time is None:
        # Полный час: считаем только заказы, которые попадают в [hour_start, hour_end).
        sub = df_orders_dt[
            (df_orders_dt["order_time"] >= hour_start) & (df_orders_dt["order_time"] < hour_end)
        ]
    else:
        # Текущий (частичный) час: считаем только заказы, которые уже "пришли" к upper_time.
        upper_time = min(upper_time, hour_end)
        sub = df_orders_dt[
            (df_orders_dt["order_time"] >= hour_start) & (df_orders_dt["order_time"] <= upper_time)
        ]
    if sub.empty:
        return 0
    if "order_id" in sub.columns:
        sub = sub.drop_duplicates(subset=["order_id"], keep="first")
    else:
        sub = sub.drop_duplicates(subset=["order_time"], keep="first")
    return int(len(sub))

# X: 08..24 (граница 24:00 будет продублирована последним значением часа 23:00)
hours_full = list(range(8, 24))  # 8..23
x_vals = list(range(8, 25))  # 8..24

history_counts = []
today_counts = []

for h in hours_full:
    # Прошлая неделя
    if pre_data_bench is not None and bench_hour_col is not None and bench_orders_col is not None:
        hist_val = int(pre_data_bench.loc[pre_data_bench[bench_hour_col] == h, bench_orders_col].sum())
    else:
        hist_val = 0
    history_counts.append(hist_val)

    # Сегодня: до текущего часа считаем полный интервал,
    # а для текущего часа - только то, что уже пришло до sim.current_time.
    if h < current_hour:
        today_counts.append(_count_orders_in_hour(today_date, h))
    elif h == current_hour:
        today_counts.append(_count_orders_in_hour(today_date, h, upper_time=sim.current_time))
    else:
        today_counts.append(None)

history_counts_ext = history_counts + ([history_counts[-1]] if history_counts else [0])
today_counts_ext = today_counts + ([today_counts[-1]] if (today_counts and current_hour >= 23) else [None])

fig_compare_orders = go.Figure()
fig_compare_orders.add_trace(
    go.Scatter(
        x=x_vals,
        y=history_counts_ext,
        mode="lines+markers",
        name="Заказы за прошлую неделю",
        line=dict(color="#FF6B6B"),
    )
)
fig_compare_orders.add_trace(
    go.Scatter(
        x=x_vals,
        y=today_counts_ext,
        mode="lines+markers",
        name="Заказы сегодня",
        line=dict(color="#96CEB4"),
    )
)
fig_compare_orders.update_layout(
    title="Заказы по часам: сегодня vs прошлой неделе",
    height=260,
    xaxis=dict(title="Час", tickmode="array", tickvals=list(range(8, 25, 1))),
    yaxis_title="Количество заказов",
    margin=dict(l=10, r=10, t=45, b=10),
)
st.plotly_chart(fig_compare_orders, width="stretch")


# Автозапуск: следующий шаг только после ответа LLM (если LLM включён)
if st.session_state.running and not sim.is_finished():
    if st.session_state.llm_enabled and st.session_state.waiting_for_llm and st.session_state.pending_llm_payload is not None:
        # Ждём ответа — повторяем запрос с тем же payload
        llm_data = call_llm_analysis(st.session_state.pending_llm_payload, st.session_state.llm_service_url)
        if llm_data.get("ok"):
            st.session_state.waiting_for_llm = False
            pending = st.session_state.pending_llm_payload or {}
            ts = (pending.get("timestamp") or "")[:19].replace("T", " ")
            st.session_state.pending_llm_payload = None
            resp_text = llm_data.get("response", "").strip()
            resp_buttons = llm_data.get("buttons", [])
            if resp_text:
                st.session_state.llm_chat_history.append({"time": ts, "response": resp_text, "buttons": resp_buttons})
            st.session_state.last_llm_response = resp_text
            st.session_state.last_llm_timestamp = ts
            st.session_state.last_llm_buttons = resp_buttons
            if resp_text:
                st.session_state.running = False
        else:
            # Если агент недоступен — показываем сообщение и ставим симуляцию на паузу.
            pending = st.session_state.pending_llm_payload or {}
            ts = (pending.get("timestamp") or "")[:19].replace("T", " ")
            fail_text = "Агент не отвечает"
            st.session_state.last_llm_response = fail_text
            st.session_state.last_llm_timestamp = ts
            st.session_state.last_llm_buttons = []
            st.session_state.llm_chat_history.append({"time": ts, "response": fail_text, "buttons": []})
            st.session_state.running = False
            st.session_state.waiting_for_llm = True
    else:
        # Делаем шаг и запрашиваем LLM
        m = compute_metrics_effective(df_orders, sim.current_time, interval_min)
        m['time'] = sim.current_time
        st.session_state.metric_history.append(m)
        payload = build_external_payload(sim.current_time, m, sim.sensors, sim.ingredient_units, sim.amount_per_pizza)
        send_sensors_to_external_system(payload)
        if st.session_state.llm_enabled:
            llm_data = call_llm_analysis(payload, st.session_state.llm_service_url)
            ts = payload.get("timestamp", "")[:19].replace("T", " ")
            if llm_data.get("ok"):
                resp_text = llm_data.get("response", "").strip()
                resp_buttons = llm_data.get("buttons", [])
                if resp_text:
                    st.session_state.llm_chat_history.append({"time": ts, "response": resp_text, "buttons": resp_buttons})
                    st.session_state.last_llm_response = resp_text
                    st.session_state.last_llm_timestamp = ts
                    st.session_state.last_llm_buttons = resp_buttons
                    st.session_state.running = False
                st.session_state.waiting_for_llm = False
                st.session_state.pending_llm_payload = None
            else:
                # Ошибка сети/таймаут/не 200 — ставим на паузу и ждём ручного повтора.
                fail_text = "Агент не отвечает"
                st.session_state.last_llm_response = fail_text
                st.session_state.last_llm_timestamp = ts
                st.session_state.last_llm_buttons = []
                st.session_state.llm_chat_history.append({"time": ts, "response": fail_text, "buttons": []})
                st.session_state.running = False
                st.session_state.waiting_for_llm = True
                st.session_state.pending_llm_payload = payload
        else:
            st.session_state.waiting_for_llm = False
            st.session_state.pending_llm_payload = None
    sim.step(interval_min)
    time.sleep(1.2 / st.session_state.speed)
    st.rerun()

if sim.is_finished():
    st.session_state.running = False
    st.balloons()
    st.success("✅ Симуляция завершена.")