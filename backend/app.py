import sys, os

# Optimize CPU threading for ML libraries on Intel multi-core CPUs
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pyrefly: ignore [missing-import]
import face_recognition
from flask import Flask, request, jsonify, send_from_directory, send_file, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
import logging

from database.models import init_db, Student, User
from models.face_engine import FaceRecognitionEngine, CameraManager
from utils.attendance_manager import AttendanceManager
from utils.registration import RegistrationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config['SECRET_KEY'] = 'face-attendance-secret-2024'
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Admin & Liveness Configurations ────────────────────────────────────────
ADMIN_ID = os.environ.get('ADMIN_ID', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'Admin@123')

# ── Database ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'attendance.db')
PHOTOS_DIR = os.path.join(PROJECT_ROOT, 'database', 'photos')
engine_db, SessionFactory = init_db(DB_PATH)

# Seed default admin user if missing
with SessionFactory() as s:
    if s.query(User).filter_by(username=ADMIN_ID).first() is None:
        default_admin = User(
            username=ADMIN_ID,
            password=User.hash_password(ADMIN_PASSWORD)
        )
        s.add(default_admin)
        s.commit()
        logger.info(f"🔑 Seeded '{ADMIN_ID}' admin user account in database.")
    if s.query(User).filter_by(username='admin').first() is None:
        admin_acc = User(
            username='admin',
            password=User.hash_password('Admin@123')
        )
        s.add(admin_acc)
        s.commit()
        logger.info("🔑 Seeded fallback 'admin' account in database.")

# ── ML Engine ─────────────────────────────────────────────────────────────
face_engine = FaceRecognitionEngine(
    tolerance=0.50,
    process_every_n=3
)
camera_mgr  = CameraManager(face_engine, camera_index=0, processing_fps=10)

attendance_mgr  = AttendanceManager(SessionFactory)
registration_mgr = RegistrationManager(SessionFactory, face_engine, photos_dir=PHOTOS_DIR)

# Pre-load known faces from DB
with SessionFactory() as s:
    face_engine.load_known_faces(s.query(Student).filter_by(is_active=True).all())


# ── Attendance callback (called from recognition thread) ──────────────────
def on_face_recognized(student_id, name, confidence):
    try:
        result = attendance_mgr.mark_attendance(student_id, name, confidence)
        if not result:
            return
        # Push real-time update to all connected browser clients
        socketio.emit('attendance_update', {
            'action': result['action'],
            'record': result.get('record'),
            'message': result.get('message'),
            'stats': attendance_mgr.get_stats()
        })
        if result['action'] != 'rejected':
            logger.info(f"🎯 {result['action'].upper()}: {name} ({confidence*100:.1f}%)")
        else:
            logger.info(f"⚠️ REJECTED: {name} ({result['message']})")
    except Exception as e:
        logger.error(f"Attendance error: {e}")

face_engine.on_recognized_callback = on_face_recognized


# ── Video Streaming Thread ─────────────────────────────────────────────────
def stream_frames():
    """Continuously push annotated frames to browser via SocketIO."""
    while True:
        if camera_mgr.running:
            frame_b64 = camera_mgr.get_frame()
            if frame_b64:
                socketio.emit('video_frame', {
                    'frame': frame_b64,
                    'fps': round(camera_mgr.fps, 1)
                })
        time.sleep(0.04)   # ~25 fps max to browser

stream_thread = threading.Thread(target=stream_frames, daemon=True)
stream_thread.start()


# ══════════════════════════════════════════════════════════════════════════
# REST API Routes
# ══════════════════════════════════════════════════════════════════════════

# ── Authentication Interceptor ──────────────────────────────────────────────
@app.before_request
def require_login():
    if request.path.startswith('/api/'):
        if request.path in ['/api/auth/login', '/api/auth/status', '/api/auth/register']:
            return None
        if not session.get('logged_in'):
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    return None

# ── Authentication Routes ──────────────────────────────────────────────────
@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    with SessionFactory() as s:
        user = s.query(User).filter_by(username=username).first()
        if user and user.verify_password(password):
            session['logged_in'] = True
            session['username'] = username
            return jsonify({'success': True, 'message': 'Logged in successfully'})
    return jsonify({'success': False, 'message': 'Invalid Admin ID or Password'}), 401

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Admin ID and password are required'}), 400
        
    # Enforce password complexity criteria (min 8 chars, uppercase, digit, special character)
    if len(password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'}), 400
    if not any(c.isupper() for c in password):
        return jsonify({'success': False, 'message': 'Password must contain at least one uppercase letter'}), 400
    if not any(c.isdigit() for c in password):
        return jsonify({'success': False, 'message': 'Password must contain at least one number'}), 400
    if not any(not c.isalnum() for c in password):
        return jsonify({'success': False, 'message': 'Password must contain at least one special character'}), 400
        
    try:
        with SessionFactory() as s:
            existing_user = s.query(User).filter_by(username=username).first()
            if existing_user:
                return jsonify({'success': False, 'message': 'Admin ID already exists'}), 409
                
            new_user = User(
                username=username,
                password=User.hash_password(password)
            )
            s.add(new_user)
            s.commit()
            return jsonify({'success': True, 'message': f'Admin user {username} registered successfully'})
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'Failed to create account due to server error'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/auth/status', methods=['GET'])
def api_auth_status():
    if session.get('logged_in'):
        return jsonify({'authenticated': True, 'username': session.get('username')})
    return jsonify({'authenticated': False})

# ── Serve Frontend ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


# ── Camera Control ─────────────────────────────────────────────────────────
@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    ok = camera_mgr.start()
    return jsonify({'success': ok, 'message': 'Camera started' if ok else 'Failed to start camera'})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    camera_mgr.stop()
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    return jsonify({'running': camera_mgr.running, 'fps': round(camera_mgr.fps, 1)})


# ── Liveness and Recognition Settings ──────────────────────────────────────
@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify({
        'success': True,
        'laplacian_threshold': face_engine.liveness_detector.texture.laplacian_threshold,
        'glare_threshold': face_engine.liveness_detector.texture.glare_threshold,
        'tolerance': face_engine.tolerance,
        'scale_factor': face_engine.scale_factor
    })

@app.route('/api/settings', methods=['POST'])
def save_settings():
    data = request.get_json() or {}
    
    # Update laplacian threshold
    lap_thresh = data.get('laplacian_threshold')
    if lap_thresh is not None:
        try:
            face_engine.liveness_detector.texture.laplacian_threshold = float(lap_thresh)
        except ValueError:
            pass
            
    # Update glare threshold
    glare_thresh = data.get('glare_threshold')
    if glare_thresh is not None:
        try:
            face_engine.liveness_detector.texture.glare_threshold = float(glare_thresh)
        except ValueError:
            pass
            
    # Update tolerance (strictness)
    tolerance = data.get('tolerance')
    if tolerance is not None:
        try:
            face_engine.tolerance = float(tolerance)
        except ValueError:
            pass
            
    # Update scale factor (resolution scale)
    scale_factor = data.get('scale_factor')
    if scale_factor is not None:
        try:
            face_engine.scale_factor = float(scale_factor)
        except ValueError:
            pass
            
    return jsonify({
        'success': True,
        'message': 'Settings updated successfully',
        'settings': {
            'laplacian_threshold': face_engine.liveness_detector.texture.laplacian_threshold,
            'glare_threshold': face_engine.liveness_detector.texture.glare_threshold,
            'tolerance': face_engine.tolerance,
            'scale_factor': face_engine.scale_factor
        }
    })


# ── Student Registration ───────────────────────────────────────────────────
@app.route('/api/students/register', methods=['POST'])
def register_student():
    try:
        student_id = request.form.get('student_id', '').strip()
        name       = request.form.get('name', '').strip()
        email      = request.form.get('email', '').strip()
        department = request.form.get('department', '').strip()
        batch      = request.form.get('batch', '').strip()

        if not all([student_id, name]):
            return jsonify({'success': False, 'message': 'student_id and name are required'}), 400

        if 'photo' not in request.files:
            return jsonify({'success': False, 'message': 'Photo file is required'}), 400

        image_bytes = request.files['photo'].read()
        success, msg, student = registration_mgr.register_student(
            student_id, name, email, department, batch, image_bytes
        )
        return jsonify({'success': success, 'message': msg, 'student': student})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/students', methods=['GET'])
def list_students():
    students = registration_mgr.get_all_students()
    return jsonify({'success': True, 'students': students, 'total': len(students)})


@app.route('/api/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    ok, msg = registration_mgr.delete_student(student_id)
    return jsonify({'success': ok, 'message': msg})


# ── Attendance Records ─────────────────────────────────────────────────────
@app.route('/api/attendance/today', methods=['GET'])
def today_attendance():
    records = attendance_mgr.get_today_attendance()
    return jsonify({'success': True, 'records': records, 'total': len(records)})


@app.route('/api/attendance/date/<date_str>', methods=['GET'])
def attendance_by_date(date_str):
    records = attendance_mgr.get_attendance_by_date(date_str)
    return jsonify({'success': True, 'records': records})


@app.route('/api/attendance/student/<student_id>', methods=['GET'])
def student_history(student_id):
    records = attendance_mgr.get_student_history(student_id)
    return jsonify({'success': True, 'records': records})


@app.route('/api/attendance/stats', methods=['GET'])
def get_stats():
    return jsonify({'success': True, 'stats': attendance_mgr.get_stats()})


@app.route('/api/attendance/weekly', methods=['GET'])
def weekly_trend():
    return jsonify({'success': True, 'trend': attendance_mgr.get_weekly_trend()})


@app.route('/api/attendance/export', methods=['GET'])
def export_attendance():
    date_str = request.args.get('date')
    path = attendance_mgr.export_csv(date_str)
    if not path:
        return jsonify({'success': False, 'message': 'No records found'}), 404
    return send_file(path, as_attachment=True)


# ── Manual Attendance ──────────────────────────────────────────────────────
@app.route('/api/attendance/manual', methods=['POST'])
def manual_attendance():
    data = request.get_json()
    student_id = data.get('student_id')
    name = data.get('name', '')
    try:
        result = attendance_mgr.mark_attendance(student_id, name, 1.0)
        socketio.emit('attendance_update', {
            'action': result['action'],
            'record': result.get('record'),
            'stats': attendance_mgr.get_stats()
        })
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ── SocketIO Events ────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    if not session.get('logged_in'):
        logger.warning("Rejected unauthorized WebSocket connection attempt.")
        return False
    emit('connected', {'message': 'Connected to Smart Attendance System'})
    emit('stats_update', attendance_mgr.get_stats())

@socketio.on('request_stats')
def on_request_stats():
    emit('stats_update', attendance_mgr.get_stats())


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print("\n" + "="*60)
    print("  [*]  Smart Face Attendance System")
    print("  [>]  http://localhost:5000")
    print("="*60 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
