# auth.py
import mysql.connector
import bcrypt
import streamlit as st

# Update with your local MySQL credentials
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234', 
    'database': 'planet_app'
}

def get_connection():
    """Establishes a connection to the MySQL database."""
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        return None

def create_user(username, email, plain_password):
    """Hashes the password and saves the new user to MySQL."""
    conn = get_connection()
    if not conn: return False

    # Hash the password securely
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)"
        cursor.execute(sql, (username, email, hashed.decode('utf-8')))
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        # This catches duplicate usernames
        return False
    finally:
        cursor.close()
        conn.close()

def verify_login(username, plain_password):
    """Fetches the user from MySQL and verifies the hashed password."""
    conn = get_connection()
    if not conn: return False

    cursor = conn.cursor(dictionary=True)
    sql = "SELECT * FROM users WHERE username = %s"
    cursor.execute(sql, (username,))
    user = cursor.fetchone()
    
    cursor.close()
    conn.close()

    if user:
        # Compare the provided password against the stored hash
        if bcrypt.checkpw(plain_password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return True
    return False