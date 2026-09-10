"""
config.py
---------
Loads all configuration from environment variables (see .env.example).
Keeping every setting in one place makes it easy to see, at a glance,
exactly what this application needs to run.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# MySQL connection
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "business_intelligence")

# Gemini API (AI Copilot)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Sessions / security
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-this-in-production")

# Uploads
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
