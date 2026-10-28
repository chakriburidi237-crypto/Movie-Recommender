# Movie Recommender AI

> **"Discover Next. Watch Smarter. Powered by Content-Based Machine Learning."**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E.svg)](https://scikit-learn.org/)

Movie Recommender AI is a high-performance, full-stack recommendation engine and web application that delivers instant, content-aware movie suggestions. By analyzing textual plot semantics and multi-label genre associations through TF-IDF feature extraction and cosine similarity, it solves the cold-start recommendation problem without relying on user tracking, invasive cookies, or sparse rating matrices.

---

## 🚀 Key Features

1. **Content-Based Similarity Engine**: Computes high-dimensional vector representations across movie plots and weighted genre tokens using Scikit-Learn TF-IDF vectorization and pairwise Cosine Similarity.
2. **Cold-Start Resilient**: Recommends top-tier matches immediately from catalog features without requiring historical user ratings or lengthy collaborative filtering warm-up periods.
3. **High-Speed FastAPI Backend**: Asynchronous REST API endpoints engineered for sub-millisecond similarity lookups, catalog pagination, dynamic search, and catalog health diagnostics.
4. **Interactive Modern Web Interface**: Clean, responsive frontend featuring instant search-as-you-type, genre tags, similarity confidence percentages, and dynamic modal detail views.
5. **Secure Authentication & Session Management**: Built-in credential handling supporting email/phone identifier authentication, salted PBKDF2-HMAC-SHA256 password hashing (120,000 iterations), and HttpOnly cookie-based session management.
6. **Curated Catalog & Metrics Summary**: Real-time summary endpoint reporting catalog size, genre distributions, and metadata statistics.
7. **Offline-Ready & Self-Contained**: Operates with zero external API dependencies or cloud databases—ready to run instantly locally or in air-gapped environments.

---

## 🛠 Tech Stack

- **Backend Framework**: FastAPI (ASGI with Uvicorn)
- **Machine Learning & Data**: Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`), Pandas, NumPy
- **Language**: Python 3.10+
- **Frontend**: Vanilla JavaScript (ES6+), HTML5, CSS3 with modern glassmorphic UI cards and responsive grid layouts
- **Security**: PBKDF2-HMAC-SHA256 password hashing, secure cookie session storage
- **Testing**: Pytest, HTTPX test client

---

## 🏗 Architecture & Directory Structure

```
Movie-Recommender/
├── data/
│   └── movies.csv            # Curated catalog (title, genres, plot descriptions)
├── static/
│   └── index.html            # Responsive single-page web application
├── tests/
│   └── test_recommender.py   # Automated unit and similarity engine tests
├── app.py                    # FastAPI application, auth, and API routes
├── recommend.py              # MovieRecommender TF-IDF ML vectorization class
├── requirements.txt          # Production and testing dependencies
├── .gitattributes            # Repository language attribution rules
├── .gitignore                # Python and environment ignore rules
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## ⚙️ How the Recommendation Algorithm Works

1. **Feature Synthesis**: Plot descriptions and genre tokens are tokenized, with genres weighted by repetition factor $W=3$ to emphasize thematic alignment.
2. **TF-IDF Vectorization**: Generates n-gram token matrices ($1 \le n \le 2$) removing standard English stop words.
3. **Cosine Similarity Matrix**: Computes pairwise cosine similarity between normalized vector coordinates:
   $$\text{Similarity}(u, v) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
4. **Ranking & Filtering**: Filters the query movie and ranks the top $K$ nearest neighbors sorted by descending similarity coefficient.

---

## ⚡ Getting Started

### Prerequisites

- Python `3.10`, `3.11`, or `3.12`
- `pip` package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/chakriburidi237-crypto/Movie-Recommender.git
   cd Movie-Recommender
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   uvicorn app:app --reload --host 127.0.0.1 --port 8000
   ```
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

---

## 🧪 Automated Testing

Automated unit tests validate recommender initialization, metadata parsing, TF-IDF calculation, and recommendation top-K ranking:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_recommender.py::test_recommender_initialization PASSED
tests/test_recommender.py::test_recommendation_logic PASSED
tests/test_recommender.py::test_summary PASSED
======================== 3 passed in 0.42s ========================
```

---

## 🚀 Deployment

### Deploy to Render / Railway / Fly.io

1. Connect your GitHub repository to [Render](https://render.com) or [Railway](https://railway.app).
2. Set Build Command: `pip install -r requirements.txt`
3. Set Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📄 License and Copyright\n\nAll rights reserved © 2026 Chakradhar Buridi.

