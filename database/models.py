from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os

Base = declarative_base()

class Student(Base):
    """Student registration table"""
    __tablename__ = 'students'

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(150))
    department = Column(String(100))
    batch = Column(String(50))
    face_encoding = Column(Text)          # JSON-serialized 128-d vector
    photo_path = Column(String(255))
    registered_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    def set_encoding(self, encoding_array):
        self.face_encoding = json.dumps(encoding_array.tolist())

    def get_encoding(self):
        import numpy as np
        if self.face_encoding:
            try:
                return np.array(json.loads(self.face_encoding), dtype=np.float64)
            except Exception:
                return None
        return None

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'name': self.name,
            'batch': self.batch,
            'email': self.email,
            'department': self.department,
            'photo_path': self.photo_path,
            'registered_at': self.registered_at.isoformat() if self.registered_at else None,
            'is_active': self.is_active
        }


class AttendanceRecord(Base):
    """Attendance log table"""
    __tablename__ = 'attendance'

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), nullable=False)
    student_name = Column(String(100))
    date = Column(String(20))             # YYYY-MM-DD
    check_in_time = Column(DateTime)
    check_out_time = Column(DateTime)
    confidence = Column(Float)
    status = Column(String(20), default='Present')  # Present / Late / Absent
    session = Column(String(50))          # Morning / Afternoon

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'student_name': self.student_name,
            'date': self.date,
            'check_in_time': self.check_in_time.isoformat() if self.check_in_time else None,
            'check_out_time': self.check_out_time.isoformat() if self.check_out_time else None,
            'confidence': round(self.confidence * 100, 1) if self.confidence else 0,
            'status': self.status,
            'session': self.session
        }


class User(Base):
    """Administrator table for system login"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(128), nullable=False)
    registered_at = Column(DateTime, default=datetime.utcnow)

    @staticmethod
    def hash_password(password):
        import hashlib
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def verify_password(self, password):
        return self.password == self.hash_password(password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'registered_at': self.registered_at.isoformat() if self.registered_at else None
        }


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'attendance.db')

def init_db(db_path=None):
    """Initialize the SQLite database with multi-thread concurrency support"""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    elif not os.path.isabs(db_path):
        db_path = os.path.join(PROJECT_ROOT, db_path)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(
        f'sqlite:///{db_path}',
        connect_args={'check_same_thread': False},
        pool_pre_ping=True,
        echo=False
    )
    Base.metadata.create_all(engine)
    
    # Auto-migration for batch column
    from sqlalchemy import inspect
    inspector = inspect(engine)
    if 'students' in inspector.get_table_names():
        columns = [c['name'] for c in inspector.get_columns('students')]
        if 'batch' not in columns:
            with engine.begin() as conn:
                conn.exec_driver_sql("ALTER TABLE students ADD COLUMN batch VARCHAR(50)")
                
    Session = sessionmaker(bind=engine)
    return engine, Session
