# auth.py
import os
import sqlite3
import bcrypt
import streamlit as st

DB_PATH = os.getenv("AUTH_DB_PATH", "auth.db")


def get_connection():
    """Opens a connection to the local SQLite auth database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the users table if it doesn't exist yet."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()


def create_user(username, email, plain_password):
    """Hashes the password and saves the new user to SQLite."""
    conn = get_connection()
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, hashed.decode("utf-8")),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Duplicate username
        return False
    except sqlite3.Error as err:
        st.error(f"Database Error: {err}")
        return False
    finally:
        conn.close()


def verify_login(username, plain_password):
    """Fetches the user from SQLite and verifies the hashed password."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()

    if row:
        if bcrypt.checkpw(plain_password.encode("utf-8"), row["password_hash"].encode("utf-8")):
            return True
    return False
