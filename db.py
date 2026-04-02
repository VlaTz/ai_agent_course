# -*- coding: utf-8 -*-
"""Работа с БД: заказы и ингредиенты."""
import sqlite3
import pandas as pd
from datetime import timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "pizzeria.db"

DEFAULT_INGREDIENTS = {
    "dough": 80,
    "cheese": 60,
    "sauce": 50,
    "pepperoni": 45,
    "mushrooms": 35,
}

# Единицы измерения по умолчанию для каждого ингредиента
DEFAULT_INGREDIENT_UNITS = {
    "dough": "шт",
    "cheese": "кг",
    "sauce": "кг",
    "pepperoni": "кг",
    "mushrooms": "кг",
}

# Расход на одну пиццу (в единицах ингредиента: шт для теста, кг для остальных)
DEFAULT_AMOUNT_PER_PIZZA = {
    "dough": 1,
    "cheese": 0.15,
    "sauce": 0.1,
    "pepperoni": 0.05,
    "mushrooms": 0.03,
}


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db(conn=None):
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_time TEXT NOT NULL,
                order_id INTEGER,
                order_type TEXT NOT NULL,
                products_count INTEGER NOT NULL DEFAULT 1,
                cooking_min REAL NOT NULL DEFAULT 0,
                shelf_min REAL NOT NULL DEFAULT 0,
                delivery_min REAL NOT NULL DEFAULT 0,
                courier_trip_min REAL NOT NULL DEFAULT 0,
                completion_time TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingredients (
                name TEXT PRIMARY KEY,
                value INTEGER NOT NULL DEFAULT 0,
                unit TEXT NOT NULL DEFAULT 'шт',
                amount_per_pizza REAL NOT NULL DEFAULT 0
            )
        """)
        # Миграция: добавить колонку unit, если её нет (старая схема)
        cur = conn.execute("PRAGMA table_info(ingredients)")
        columns = [row[1] for row in cur.fetchall()]
        if "unit" not in columns:
            conn.execute("ALTER TABLE ingredients ADD COLUMN unit TEXT NOT NULL DEFAULT 'шт'")
            for name in DEFAULT_INGREDIENTS:
                unit = DEFAULT_INGREDIENT_UNITS.get(name, "шт")
                conn.execute("UPDATE ingredients SET unit = ? WHERE name = ?", (unit, name))
        if "amount_per_pizza" not in columns:
            conn.execute("ALTER TABLE ingredients ADD COLUMN amount_per_pizza REAL NOT NULL DEFAULT 0")
            for name in DEFAULT_INGREDIENTS:
                amt = DEFAULT_AMOUNT_PER_PIZZA.get(name, 0)
                conn.execute("UPDATE ingredients SET amount_per_pizza = ? WHERE name = ?", (amt, name))
        conn.commit()
        cur = conn.execute("SELECT COUNT(*) FROM ingredients")
        if cur.fetchone()[0] == 0:
            for name, value in DEFAULT_INGREDIENTS.items():
                unit = DEFAULT_INGREDIENT_UNITS.get(name, "шт")
                amt = DEFAULT_AMOUNT_PER_PIZZA.get(name, 0)
                conn.execute(
                    "INSERT INTO ingredients (name, value, unit, amount_per_pizza) VALUES (?, ?, ?, ?)",
                    (name, value, unit, amt),
                )
            conn.commit()
    finally:
        if close:
            conn.close()


def import_orders_from_excel(excel_path, parse_time_to_minutes):
    """Импорт заказов из data.xlsx в таблицу orders. Очищает таблицу перед импортом."""
    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]
    col_dt = df.columns[0]
    col_order_id = df.columns[1]
    col_type = df.columns[2]
    col_products = df.columns[3]
    col_cook = df.columns[4]
    col_shelf = df.columns[5]
    col_delivery = df.columns[6]
    col_courier = df.columns[7]

    df["order_time"] = pd.to_datetime(df[col_dt])
    raw_type = df[col_type].astype(str).str.strip().str.lower()
    df["order_type"] = raw_type.apply(lambda x: "del" if "доставка" in str(x) else "rest")
    df["cooking_min"] = df[col_cook].apply(parse_time_to_minutes)
    df["shelf_min"] = df[col_shelf].apply(parse_time_to_minutes)
    df["delivery_min"] = df[col_delivery].apply(parse_time_to_minutes)
    df["courier_trip_min"] = df[col_courier].apply(parse_time_to_minutes)
    df["products_count"] = pd.to_numeric(df[col_products], errors="coerce").fillna(1).astype(int).clip(lower=1)
    df["completion_time"] = df["order_time"] + df["cooking_min"].apply(lambda m: timedelta(minutes=m))
    df["completion_time"] = df["completion_time"] + df["shelf_min"].apply(lambda m: timedelta(minutes=m))
    df["completion_time"] = df["completion_time"] + df.apply(
        lambda r: timedelta(minutes=r["delivery_min"]) if r["order_type"] == "del" else timedelta(0), axis=1
    )

    conn = get_connection()
    try:
        conn.execute("DELETE FROM orders")
        for _, row in df.iterrows():
            conn.execute(
                """INSERT INTO orders (order_time, order_id, order_type, products_count,
                   cooking_min, shelf_min, delivery_min, courier_trip_min, completion_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["order_time"].isoformat(),
                    int(row.get(col_order_id, 0)),
                    row["order_type"],
                    int(row["products_count"]),
                    float(row["cooking_min"]),
                    float(row["shelf_min"]),
                    float(row["delivery_min"]),
                    float(row["courier_trip_min"]),
                    row["completion_time"].isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_orders_df():
    """Загружает заказы из БД в DataFrame с колонками как в приложении."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT order_time, order_id, order_type, products_count, "
            "cooking_min, shelf_min, delivery_min, courier_trip_min, completion_time FROM orders ORDER BY order_time",
            conn,
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["order_time"] = pd.to_datetime(df["order_time"])
    df["completion_time"] = pd.to_datetime(df["completion_time"])
    return df


def get_ingredients():
    """Возвращает список словарей [{"name", "value", "unit", "amount_per_pizza"}, ...] из БД."""
    conn = get_connection()
    try:
        cur = conn.execute("PRAGMA table_info(ingredients)")
        cols = [row[1] for row in cur.fetchall()]
        if "amount_per_pizza" in cols:
            cur = conn.execute("SELECT name, value, unit, amount_per_pizza FROM ingredients")
            return [
                {"name": row[0], "value": int(row[1]), "unit": (row[2] or "шт"), "amount_per_pizza": float(row[3]) if row[3] is not None else DEFAULT_AMOUNT_PER_PIZZA.get(row[0], 0)}
                for row in cur.fetchall()
            ]
        cur = conn.execute("SELECT name, value, unit FROM ingredients")
        return [
            {"name": row[0], "value": int(row[1]), "unit": (row[2] or "шт"), "amount_per_pizza": DEFAULT_AMOUNT_PER_PIZZA.get(row[0], 0)}
            for row in cur.fetchall()
        ]
    finally:
        conn.close()


def save_ingredients(sensors_dict):
    """Сохраняет текущие значения ингредиентов в БД (только value, unit не меняется)."""
    conn = get_connection()
    try:
        for name, value in sensors_dict.items():
            if name in DEFAULT_INGREDIENTS:
                cur = conn.execute("UPDATE ingredients SET value = ? WHERE name = ?", (max(0, int(value)), name))
                if cur.rowcount == 0:
                    unit = DEFAULT_INGREDIENT_UNITS.get(name, "шт")
                    amt = DEFAULT_AMOUNT_PER_PIZZA.get(name, 0)
                    conn.execute(
                        "INSERT INTO ingredients (name, value, unit, amount_per_pizza) VALUES (?, ?, ?, ?)",
                        (name, max(0, int(value)), unit, amt),
                    )
        conn.commit()
    finally:
        conn.close()
