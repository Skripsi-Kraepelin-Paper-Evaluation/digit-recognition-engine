from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class KraepelinProject(Base):
    __tablename__ = 'kraepelin_projects'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    occupacy_and_role = Column(String(255))
    last_edu = Column(String(50))
    pob = Column(String(255))
    dob = Column(String(50))
    test_date = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PreviewHistory(Base):
    __tablename__ = 'preview_history'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    questions = Column(JSON, nullable=False)
    answers = Column(JSON, nullable=False)
    total_questions = Column(Integer)
    total_answers = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EvalHistory(Base):
    __tablename__ = 'eval_history'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    panker = Column(String(50))
    tianker = Column(String(50))
    janker = Column(String(50))
    jankerv2 = Column(String(50))
    hanker = Column(String(255))
    accuracy = Column(String(50))
    col_score_per_minute = Column(String(50))
    total_correct_ans = Column(String(50))
    plot_image_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
