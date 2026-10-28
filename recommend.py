"""
Content-based movie recommender.

Builds a TF-IDF representation over each movie's genres + description
(genres weighted more heavily) and recommends the most similar movies
via cosine similarity — no user ratings required.
"""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "movies.csv"


class MovieRecommender:
    def __init__(self, data_path: Path = DATA_PATH):
        self.df = pd.read_csv(data_path)
        required_columns = {"title", "genres", "description"}
        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            raise ValueError(
                f"Movie catalog is missing columns: {', '.join(sorted(missing_columns))}"
            )

        self.df["genres"] = self.df["genres"].fillna("").astype(str)
        self.df["description"] = self.df["description"].fillna("").astype(str)
        # Weight genres more heavily than description by repeating them
        self.df["combined"] = (
            (self.df["genres"] + " ") * 3 + self.df["description"]
        ).str.lower()

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined"])
        self.similarity = cosine_similarity(self.tfidf_matrix)

        self.title_to_idx = {
            title.lower(): idx for idx, title in enumerate(self.df["title"])
        }

    def titles(self):
        return self.df["title"].tolist()

    def summary(self):
        genres = {
            genre.strip()
            for value in self.df["genres"]
            for genre in value.split()
            if genre.strip()
        }
        return {
            "movie_count": len(self.df),
            "genre_count": len(genres),
            "model": "TF-IDF + cosine similarity",
            "features": "Genres (3x weight) + plot description",
        }

    def recommend(self, title: str, top_n: int = 5):
        key = title.strip().lower()
        if key not in self.title_to_idx:
            # fuzzy fallback: substring match
            matches = [t for t in self.df["title"] if key in t.lower()]
            if not matches:
                raise ValueError(f"Movie '{title}' not found in catalog.")
            key = matches[0].lower()

        idx = self.title_to_idx[key]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        results = []
        source_genres = {
            genre.strip().lower() for genre in self.df.iloc[idx]["genres"].split()
        }
        for i, score in scores:
            row = self.df.iloc[i]
            matched_genres = [
                genre for genre in row["genres"].split()
                if genre.strip().lower() in source_genres
            ]
            results.append({
                "title": row["title"],
                "genres": row["genres"],
                "description": row["description"],
                "similarity": round(float(score), 4),
                "matched_genres": matched_genres,
                "reason": (
                    f"Shares {', '.join(matched_genres)} themes"
                    if matched_genres
                    else "Similar plot language"
                ),
            })
        return results


if __name__ == "__main__":
    rec = MovieRecommender()
    demo_title = "Neon Horizon"
    print(f"Recommendations for '{demo_title}':\n")
    for r in rec.recommend(demo_title, top_n=5):
        print(f"  {r['similarity']:.3f}  {r['title']}  ({r['genres']})")
