import os
import shutil
from datetime import datetime
from database.models import Student


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PHOTOS_DIR = os.path.join(PROJECT_ROOT, 'database', 'photos')


class RegistrationManager:
    """Handles enrolling new students into the system."""

    def __init__(self, db_session_factory, face_engine, photos_dir=None):
        self.Session = db_session_factory
        self.engine = face_engine
        if photos_dir is None:
            self.PHOTOS_DIR = DEFAULT_PHOTOS_DIR
        elif not os.path.isabs(photos_dir):
            self.PHOTOS_DIR = os.path.join(PROJECT_ROOT, photos_dir)
        else:
            self.PHOTOS_DIR = photos_dir
        os.makedirs(self.PHOTOS_DIR, exist_ok=True)

    def register_student(self, student_id, name, email, department, batch, image_bytes):
        """
        Register a new student.
        Returns (success: bool, message: str, student_dict or None)
        """
        session = self.Session()
        try:
            # Check duplicate
            existing = session.query(Student).filter_by(student_id=student_id).first()
            if existing:
                return False, f"Student ID '{student_id}' already registered", None

            # Encode face
            encoding, found, msg = self.engine.encode_face_from_image(image_bytes)
            if not found:
                return False, msg, None

            # Check duplicate face similarity
            all_students = session.query(Student).filter_by(is_active=True).all()
            if all_students:
                known_encodings = []
                student_map = []
                for s in all_students:
                    enc = s.get_encoding()
                    if enc is not None:
                        known_encodings.append(enc)
                        student_map.append(s)
                
                if known_encodings:
                    import numpy as np
                    # pyrefly: ignore [missing-import]
                    import face_recognition
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    best_idx = int(np.argmin(distances))
                    best_dist = distances[best_idx]
                    
                    tolerance = getattr(self.engine, 'tolerance', 0.50)
                    if best_dist < tolerance:
                        matched = student_map[best_idx]
                        return False, f"This face is already registered with Roll No: {matched.student_id} and Name: {matched.name}.", None

            # Save photo
            photo_path = self._save_photo(student_id, image_bytes)

            # Save to DB
            student = Student(
                student_id=student_id,
                name=name,
                email=email,
                department=department,
                batch=batch,
                photo_path=photo_path,
                registered_at=datetime.utcnow()
            )
            student.set_encoding(encoding)
            session.add(student)
            session.commit()

            # Reload engine
            all_students = session.query(Student).filter_by(is_active=True).all()
            self.engine.load_known_faces(all_students)

            return True, "Student registered successfully", student.to_dict()

        except Exception as e:
            session.rollback()
            return False, str(e), None
        finally:
            session.close()

    def get_all_students(self):
        session = self.Session()
        try:
            students = session.query(Student).filter_by(is_active=True).all()
            return [s.to_dict() for s in students]
        finally:
            session.close()

    def delete_student(self, student_id):
        session = self.Session()
        try:
            s = session.query(Student).filter_by(student_id=student_id).first()
            if not s:
                return False, "Student not found"
            s.is_active = False
            session.commit()
            all_students = session.query(Student).filter_by(is_active=True).all()
            self.engine.load_known_faces(all_students)
            return True, "Student removed"
        except Exception as e:
            session.rollback()
            return False, str(e)
        finally:
            session.close()

    def _save_photo(self, student_id, image_bytes):
        filename = f"{student_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        path = os.path.join(self.PHOTOS_DIR, filename)
        with open(path, 'wb') as f:
            f.write(image_bytes)
        return path
