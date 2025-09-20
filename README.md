# 🚀 NORMSCAN v3 — ИИ-сервис для выявления КТ ОГК с «нормой»

## ✅ Описание
NORMSCAN — это **альтернативный подход** к анализу КТ органов грудной клетки:  
> **Мы не учимся распознавать 40 патологий — мы учимся чувствовать, когда чего-то НЕТ.**

Модель обучается на **идеальной синтетической норме**.  
Все реальные КТ-исследования, даже если они "нормальные", **классифицируются как патология**, потому что **отклоняются от идеала**.

## ✅ Технологии
- **Backend**: FastAPI + PyTorch + CUDA (RTX 3090 Ti)
- **Frontend**: ReactJS + Axios
- **Deployment**: Docker + docker-compose
- **Output**: Excel-отчёт по ТЗ

## ✅ Быстрый старт (Quick Start)

```bash
git clone https://github.com/your-team/normscan-final.git
cd normscan-final
docker-compose up --build
