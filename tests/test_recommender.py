import pytest
from pathlib import Path
from recommend import MovieRecommender

def test_recommender_initialization():
    recommender = MovieRecommender()
    titles = recommender.titles()
    assert len(titles) > 0
    assert "Inception" in titles or len(titles) >= 10

def test_recommendation_logic():
    recommender = MovieRecommender()
    titles = recommender.titles()
    first_title = titles[0]
    recommendations = recommender.recommend(first_title, top_k=5)
    assert len(recommendations) <= 5
    for rec in recommendations:
        assert "title" in rec
        assert "similarity" in rec
        assert "genres" in rec

def test_summary():
    recommender = MovieRecommender()
    summary = recommender.summary()
    assert summary["total_movies"] > 0
