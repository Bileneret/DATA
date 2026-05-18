import pandas as pd
import sqlite3
import requests
import os
import json
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# --- РОБИМО ШЛЯХИ БРОНЕБІЙНИМИ ---
# Дізнаємось папку, де лежить САМ ЦЕЙ СКРИПТ (тобто mini-project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Тепер всі файли і бази будуть створюватись і читатись тільки тут
API_TEMP = os.path.join(BASE_DIR, "api_temp.json")
CSV_TEMP = os.path.join(BASE_DIR, "csv_temp.json")
CSV_SOURCE = os.path.join(BASE_DIR, "users_extra.csv")
DB_PATH = os.path.join(BASE_DIR, "integration.db")

@tool
def fetch_api(url: str) -> str:
    """Вивантажує дані з API та зберігає у тимчасовий файл"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(API_TEMP, "w", encoding="utf-8") as f:
            f.write(response.text)
        return "Дані з API успішно збережено."
    except Exception as e:
        return f"Помилка API: {e}"

@tool
def fetch_csv(filepath: str) -> str:
    """Зчитує локальний CSV файл та зберігає у тимчасовий файл"""
    try:
        # Читаємо файл за абсолютним шляхом
        df = pd.read_csv(CSV_SOURCE)
        df.to_json(CSV_TEMP, orient="records")
        return "Дані з CSV успішно збережено."
    except Exception as e:
        return f"Помилка CSV: {e}"

@tool
def merge_and_load_db(table_name: str) -> str:
    """Зливає дані з тимчасових файлів по полю 'id' та зберігає в SQLite."""
    try:
        df_api = pd.read_json(API_TEMP)
        df_csv = pd.read_json(CSV_TEMP)
        
        # JOIN по id
        df_merged = pd.merge(df_api, df_csv, on="id", how="left")
        
        # Трансформація: словники/списки -> JSON-рядки
        for col in df_merged.columns:
            df_merged[col] = df_merged[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
            
        # Завантаження в БД
        conn = sqlite3.connect(DB_PATH)
        df_merged.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        
        # Замітаємо сліди
        for file in [API_TEMP, CSV_TEMP]:
            if os.path.exists(file):
                os.remove(file)
                
        return f"Успіх! Об'єднано та завантажено {len(df_merged)} записів у таблицю {table_name}."
    except Exception as e:
        return f"Помилка БД: {e}"

# 1. Реєструємо інструменти
tools = [fetch_api, fetch_csv, merge_and_load_db]

# 2. Ініціалізуємо модель
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 3. Створюємо агента
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "Ти - ETL агент. Виконай наступні кроки послідовно:\n"
        "1. Отримай дані з API.\n"
        "2. Отримай дані з локального CSV файлу.\n"
        "3. Злий їх та завантаж у базу даних.\n"
        "Поверни фінальний статус операції."
    )
)

# 4. Запуск
if __name__ == "__main__":
    task = "Отримай користувачів з https://jsonplaceholder.typicode.com/users, зчитай файл users_extra.csv і збережи об'єднані дані в таблицю final_users."
    
    print("Запуск ETL агента...")
    response = agent.invoke({"messages": [("user", task)]})
    
    print("\n--- ФІНАЛЬНИЙ РЕЗУЛЬТАТ ---")
    print(response["messages"][-1].content)