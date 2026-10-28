"""
FastAPI backend for the Movie Recommendation System.

Run:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in your browser.
"""
from pathlib import Path
import hashlib
import re
import secrets
from typing import Literal

from fastapi import Cookie, FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from recommend import MovieRecommender

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommender using TF-IDF + cosine similarity over genres and plot descriptions.",
    version="1.0.0",
)

recommender = MovieRecommender()

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_PATTERN = re.compile(r"^\+?[0-9\s().-]{8,20}$")
users = {}
sessions = {}


def hash_password(password: str, salt: bytes | None = None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return salt.hex(), digest.hex()


def password_matches(password: str, salt_hex: str, digest_hex: str):
    _, candidate = hash_password(password, bytes.fromhex(salt_hex))
    return secrets.compare_digest(candidate, digest_hex)


def display_name(identifier: str):
    if "@" in identifier:
        return identifier.split("@", 1)[0].replace(".", " ").title()
    return "Member"


def validate_identifier(identifier: str, login_type: str):
    identifier = identifier.strip()
    pattern = EMAIL_PATTERN if login_type == "email" else PHONE_PATTERN
    if not pattern.match(identifier):
        label = "email address" if login_type == "email" else "phone number"
        raise HTTPException(status_code=422, detail=f"Enter a valid {label}.")
    return identifier.lower() if login_type == "email" else identifier


def current_user(session: str | None):
    identifier = sessions.get(session)
    if not identifier or identifier not in users:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    return users[identifier]


class AuthRequest(BaseModel):
    identifier: str
    password: str
    login_type: Literal["email", "phone"] = "email"


class RegisterRequest(AuthRequest):
    name: str = ""


def add_user(identifier: str, password: str, name: str = ""):
    salt, digest = hash_password(password)
    users[identifier] = {
        "identifier": identifier,
        "name": name.strip() or display_name(identifier),
        "salt": salt,
        "password_digest": digest,
    }


add_user("demo@reelmind.ai", "ReelMind123!", "Demo Explorer")


@app.post("/auth/register")
def register(payload: RegisterRequest, response: Response):
    identifier = validate_identifier(payload.identifier, payload.login_type)
    if len(payload.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")
    if identifier in users:
        raise HTTPException(status_code=409, detail="That account already exists. Try signing in.")
    add_user(identifier, payload.password, payload.name)
    token = secrets.token_urlsafe(32)
    sessions[token] = identifier
    response.set_cookie("reelmind_session", token, httponly=True, samesite="lax", max_age=86400)
    return {"user": {"name": users[identifier]["name"], "identifier": identifier}}


@app.post("/auth/login")
def login(payload: AuthRequest, response: Response):
    identifier = validate_identifier(payload.identifier, payload.login_type)
    user = users.get(identifier)
    if not user or not password_matches(payload.password, user["salt"], user["password_digest"]):
        raise HTTPException(status_code=401, detail="Those credentials do not match our records.")
    token = secrets.token_urlsafe(32)
    sessions[token] = identifier
    response.set_cookie("reelmind_session", token, httponly=True, samesite="lax", max_age=86400)
    return {"user": {"name": user["name"], "identifier": identifier}}


@app.get("/auth/me")
def me(reelmind_session: str | None = Cookie(default=None)):
    user = current_user(reelmind_session)
    return {"name": user["name"], "identifier": user["identifier"]}


@app.post("/auth/logout")
def logout(response: Response, reelmind_session: str | None = Cookie(default=None)):
    if reelmind_session:
        sessions.pop(reelmind_session, None)
    response.delete_cookie("reelmind_session")
    return {"status": "signed_out"}


@app.get("/movies")
def list_movies(reelmind_session: str | None = Cookie(default=None)):
    current_user(reelmind_session)
    return {"titles": recommender.titles()}


@app.get("/insights")
def insights(reelmind_session: str | None = Cookie(default=None)):
    current_user(reelmind_session)
    return recommender.summary()


@app.get("/recommend")
def recommend(
    title: str = Query(..., description="Movie title to base recommendations on"),
    top_n: int = Query(5, ge=1, le=15),
    reelmind_session: str | None = Cookie(default=None),
):
    current_user(reelmind_session)
    try:
        results = recommender.recommend(title, top_n=top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"query": title, "recommendations": results}


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))
