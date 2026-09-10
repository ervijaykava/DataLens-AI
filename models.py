"""
models.py
---------
All MySQL tables used by the application, defined as plain SQLAlchemy
models. Kept deliberately small - four tables is enough for this app:

    users            one row per signed-up user
    datasets         one row per user describing their current uploaded dataset
    dataset_columns  one row per detected column in that dataset (name, type, role)
    copilot_messages history of AI Copilot questions/answers, so the chat
                     page has something to show across visits

The actual uploaded ROWS of data are not stored here - each dataset gets
its own real MySQL table (created dynamically), see data.py.
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    datasets = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("CopilotMessage", back_populates="user", cascade="all, delete-orphan")


class Dataset(Base):
    """Metadata about the ONE active dataset each user currently has loaded.
    The real row data lives in its own table, named in `table_name`."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    table_name = Column(String(64), nullable=False)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    missing_values = Column(Integer, default=0)
    duplicate_rows = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="datasets")
    columns = relationship("DatasetColumn", back_populates="dataset", cascade="all, delete-orphan")


class DatasetColumn(Base):
    """One row per column detected in the active dataset - what it's called,
    what data type it holds, and what business role it plays (revenue,
    product, region, etc.) if any."""
    __tablename__ = "dataset_columns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    column_name = Column(String(255), nullable=False)
    data_type = Column(String(50), nullable=False)
    role = Column(String(50), nullable=True)

    dataset = relationship("Dataset", back_populates="columns")


class CopilotMessage(Base):
    """One row per AI Copilot question/answer pair, so the chat has history."""
    __tablename__ = "copilot_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source = Column(String(20), default="gemini")  # "gemini" or "fallback"
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="messages")
