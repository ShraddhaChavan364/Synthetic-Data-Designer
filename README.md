# NeMo Data Designer – Synthetic Data Generator

[![Docker](https://img.shields.io/badge/Docker-blue)](https://www.docker.com/) [![Python](https://img.shields.io/badge/Python-3.9+-yellow)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)](https://streamlit.io/)

A **Dockerized FastAPI backend** + **Streamlit frontend** app for designing schemas and generating synthetic data using **NVIDIA NeMo Data Designer**.

---

## ✨ Features
- **Backend (FastAPI)**: Health checks, schema templates, validation, preview generation
- **Frontend (Streamlit)**: Interactive UI for schema design, preview, generation, and analytics
- **Docker Compose**: One-command setup for backend + frontend

---


## 📂 Project Structure
```
project-root/
├─ backend/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ main.py
│  └─ utils.py
├─ frontend/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ app.py
├─ docker-compose.yml
├─ .env.example
├─ .gitignore
├─ .dockerignore
└─ README.md
```

---

## 🔐 Environment Variables
Create `.env` in **project root** based on `.env.example`:
```
# Backend
NEMO_DD_API_KEY=your_api_key_here

# Frontend
NEMO_BACKEND_URL=http://backend:9000
```
Then:
```bash
cp .env.example .env
# Fill in your API key
```

---

## 🚀 Quick Start
```bash
# Clone the repo
git clone https://github.com/your-username/nemo-data-designer.git
cd nemo-data-designer

# Copy environment file
cp .env.example .env

# Build and run containers
docker compose up --build
```
Access:
- **Frontend** → http://localhost:8501
- **Backend** → http://localhost:9001/docs

---

##  Healthcheck
- Backend: `/health`
- Frontend waits until backend is healthy

---

##  Development Notes
- Do **NOT** commit `.env` or secrets
- Generated datasets → `outputs/` (ignored by Git)
- Ports: Backend `9001`, Frontend `8501`

---

## 📦 Requirements
**Backend**:
```
fastapi
uvicorn[standard]
python-dotenv
pandas
requests
nemo-microservices-data-designer
```
**Frontend**:
```
streamlit
pandas
requests
```

---

## 🧹 Ignore Files
```
__pycache__/
*.pyc
venv/
.env
outputs/
```

---

## 🤝 Contributing
1. Fork the repo
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push and open a Pull Request



---

##  Quick Commands
```bash
# Stop and remove containers
docker compose down -v

# Rebuild without cache
docker compose build --no-cache

# Check logs
docker logs <container_name>
```
