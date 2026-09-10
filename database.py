"""
database.py
------------
This is the ONLY place that knows how to connect to MySQL. Everything else
in the app asks for a database session through `get_db()` and never talks
to SQLAlchemy's engine directly.

The connection string is built from the MYSQL_* variables in your .env
file (see .env.example / config.py).
"""

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, declarative_base

import config

DATABASE_URL = URL.create(
    "mysql+pymysql",
    username=config.MYSQL_USER,
    password=config.MYSQL_PASSWORD,
    host=config.MYSQL_HOST,
    port=int(config.MYSQL_PORT),
    database=config.MYSQL_DATABASE,
)
# pool_pre_ping avoids "MySQL server has gone away" errors after idle periods.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields one database session per request
    and always closes it afterwards, even if the request raised an error."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
