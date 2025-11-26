# 🧹 Data Quality Studio — AI-Assisted CSV Cleaning Platform

Data Quality Studio is a full-stack web application that allows users to upload CSV files, analyze data quality, preview transformation rules, clean missing values, and download the processed dataset.  
It supports secure login, data preview, rule suggestions, and a modern SaaS-style UI.

---

## 🚀 Features

### 🔐 Authentication (JWT-based)
- User registration & login
- Protected API endpoints
- Auto-display of logged-in username & logout button

### 📤 Upload & Preview
- Drag-and-drop CSV upload
- Automatic dataset summary (missing values, unique counts, dtype, top values)
- Table preview before & after transformation

### 🧠 Smart Rule Suggestions
- Detects best-fit cleaning operations per column
- One-click apply & visual diff view

### 🩺 Data Cleaning Actions
- Parse numeric, remove whitespace, dedupe rows, lowercase all text
- Fill missing values (auto / zero strategy)
- Download cleaned CSV instantly

### 💻 Tech Stack
| Frontend | Backend | Database | Auth | Styling |
|---------|---------|-----------|-------|---------|
| React + Vite | FastAPI (Python) | SQLite | JWT | Custom Dark UI + Icons |

---

## 🖼 UI Preview

| Home + Upload | Cleaning Actions + Diff |
|--------------|------------------------|
| (insert screenshot here) | (insert screenshot here) |

---

## ⚙ Setup Instructions

### 1️⃣ Clone Repo
```bash
git clone https://github.com/PrajwinRodrigues/data-quality-studio
cd data-quality-studio
