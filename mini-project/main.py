import uvicorn
import sqlite3
import webbrowser
import threading
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

# Імпортуємо нашого готового агента з твого скрипта
from agent import agent

app = FastAPI(title="ETL Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "integration.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Переконаємось, що папка static існує
os.makedirs(STATIC_DIR, exist_ok=True)

class ETLResponse(BaseModel):
    message: str
    raw_response: str

@app.post("/api/run_etl", response_model=ETLResponse)
def run_etl_process():
    # Завдання для нашого агента
    task = "Отримай користувачів з https://jsonplaceholder.typicode.com/users, зчитай файл users_extra.csv і збережи об'єднані дані в таблицю final_users."
    
    # Запускаємо Ланцюг
    response = agent.invoke({"messages": [("user", task)]})
    
    # --- ФІКС ТУТ ---
    raw_content = response["messages"][-1].content
    
    # Перевіряємо: якщо Gemini повернув список словників, дістаємо звідти текст. Якщо рядок - залишаємо як є.
    if isinstance(raw_content, list) and len(raw_content) > 0 and 'text' in raw_content[0]:
        final_text = raw_content[0]['text']
    else:
        final_text = str(raw_content)
        
    return {
        "message": "ETL пайплайн завершено",
        "raw_response": final_text
    }

@app.get("/api/data")
def get_db_data():
    """Витягує дані з БД для таблиці на фронтенді"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM final_users")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        return {"error": str(e)}

# Роздача статики (HTML, CSS, JS) - має бути в самому кінці!
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    def open_browser():
        time.sleep(1.5)  # Даємо серверу час на старт
        webbrowser.open("http://127.0.0.1:8000")

    # Відкриваємо браузер в окремому потоці
    threading.Thread(target=open_browser).start()
    
    # Запуск сервера
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)