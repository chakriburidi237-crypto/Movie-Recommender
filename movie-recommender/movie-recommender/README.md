# Movie Recommendation System

A content-based movie recommendation engine that suggests similar movies based on genre and plot description — no user rating history required. Built as one of two required projects for the Codec Technologies 1-Month AI Internship.

## Overview

- **Approach**: Content-based filtering (TF-IDF + cosine similarity), genres weighted 3x over plot description
- **Backend**: FastAPI REST API with cookie sessions (`/auth/login`, `/auth/register`, `/recommend`, `/movies`)
- **Frontend**: Vanilla HTML/CSS/JS single page — no build step required
- **Authentication**: Email or phone number login, password hashing, protected recommendation workspace, and sign out
- **Explainability**: Each result includes shared genre signals and a visual similarity score
- **Catalog**: 48 movies across 10+ genres (Sci-Fi, Romance, Action, Horror, Comedy, Crime, Animated, Historical)

## Project Structure

```
movie-recommender/
├── data/
│   └── movies.csv        # movie catalog: title, genres, description
├── static/
│   └── index.html         # frontend UI
├── recommend.py            # MovieRecommender class (TF-IDF + cosine similarity)
├── app.py                   # FastAPI app
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000**, sign in with an email or phone number, then pick a movie from the dropdown to get the 6 most similar titles ranked by similarity score.

The interface also shows model metadata, confidence bars, and a plain-language explanation for every recommendation.

For a quick demo, use **demo@reelmind.ai** with password **ReelMind123!**. New users can create an in-memory account from the Create access tab. Accounts and sessions reset when the server restarts; use a database and an OAuth provider before production deployment.

## Quick CLI test (no server needed)

```bash
python recommend.py
```

## API

**GET** `/recommend?title=Neon Horizon&top_n=5`

```json
{
  "query": "Neon Horizon",
  "recommendations": [
    { "title": "Quantum Heist", "genres": "Sci-Fi Action", "description": "...", "similarity": 0.567 },
    { "title": "Echoes of Mars", "genres": "Sci-Fi Drama", "description": "...", "similarity": 0.265 }
  ]
}
```

**GET** `/movies` — returns the full list of titles in the catalog (used to populate the dropdown).

**GET** `/insights` — returns catalog size, genre signal count, and the active feature model for the dashboard.

**GET** `/auth/me` — returns the current signed-in user, or `401` when no session exists.

## How it works

Each movie's genres and description are combined into one text blob (genres repeated 3x to weight them more heavily), vectorized with TF-IDF (unigrams + bigrams, English stop words removed), then compared pairwise using cosine similarity. For a query title, the engine returns the N movies with the highest similarity score, excluding the query itself.

## Notes / next steps

- Swap `data/movies.csv` for a real dataset (e.g. MovieLens or TMDB 5000) to scale beyond 50 titles — no code changes needed as long as the same columns (`title`, `genres`, `description`) are present.
- To add collaborative filtering (recommendations based on what similar *users* liked, not just similar *content*), you'd need a user-ratings matrix and could layer in matrix factorization (e.g. SVD via `surprise` or `scikit-surprise`) alongside this content-based approach for a hybrid recommender.

## Tech Stack

`Python` · `FastAPI` · `scikit-learn` · `pandas` · `HTML/CSS/JS`
