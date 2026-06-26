from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    raw_text = Column(Text, nullable=False)
    normalized_skills = Column(JSON, nullable=True)   # list of strings
    embedding_id = Column(String, nullable=True)      # Qdrant point ID (string)
    created_at = Column(DateTime, server_default=func.now())

    scores = relationship("CandidateScore", back_populates="candidate")

class JobRun(Base):
    __tablename__ = "job_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    jd_text = Column(Text, nullable=False)
    filters = Column(JSON, nullable=True)             # e.g. {"location": "NY", "experience": 3}
    created_at = Column(DateTime, server_default=func.now())

    scores = relationship("CandidateScore", back_populates="job_run")

class CandidateScore(Base):
    __tablename__ = "candidate_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_run_id = Column(Integer, ForeignKey("job_runs.id"), nullable=False)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    bm25_score = Column(Float, nullable=True)
    vector_score = Column(Float, nullable=True)
    reranker_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    job_run = relationship("JobRun", back_populates="scores")
    candidate = relationship("Candidate", back_populates="scores")