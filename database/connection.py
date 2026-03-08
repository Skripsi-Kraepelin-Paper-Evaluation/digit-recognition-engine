from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
import os

db = None
SessionLocal = None

def get_database_url():
    """Get database URL from environment variables"""
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'kraepelin')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')
    
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

def init_db(app=None):
    """Initialize database connection and create tables"""
    global db, SessionLocal
    
    database_url = get_database_url()
    
    engine = create_engine(
        database_url,
        poolclass=NullPool,
        echo=False
    )
    
    # Import models to ensure they're registered
    from .models import Base, KraepelinProject, PreviewHistory, EvalHistory
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session factory
    SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    
    db = SessionLocal
    
    return db

def get_db():
    """Get database session"""
    if SessionLocal is None:
        init_db()
    return SessionLocal()

def close_db():
    """Close database session"""
    if SessionLocal:
        SessionLocal.remove()
