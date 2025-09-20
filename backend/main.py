# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.predict import router as predict_router

app = FastAPI(title="NORMSCAN - AI Service for Normal CT Detection", version="3.0")

# Настройка CORS (для React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(predict_router, prefix="/api", tags=["predict"])


@app.get("/")
def root():
    return {"message": "NORMSCAN v3 is running. Go to /docs for API docs."}
