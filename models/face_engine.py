import cv2
import numpy as np
import os
# pyrefly: ignore [missing-import]
import face_recognition
from datetime import datetime
import threading
import time
import base64
import logging
from models.liveness import CombinedLivenessDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionEngine:
    """
    Real-time multi-face recognition engine.
    Detects and recognizes multiple faces simultaneously from a camera feed.
    """

    def __init__(self, tolerance=0.50, process_every_n=3, scale_factor=0.50):
        self.tolerance = tolerance          # Lower = stricter matching
        self.process_every_n = process_every_n
        self.scale_factor = scale_factor

        # Known faces database (loaded from DB)
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []

        # Real-time state
        self.frame_count = 0
        self.last_face_locations = []
        self.last_recognized = []
        self.lock = threading.Lock()

        # Attendance cooldown: prevents marking same person twice within N seconds
        self.recent_attendance = {}       # {student_id: last_marked_timestamp}
        self.cooldown_seconds = 30

        # Callback for when someone is recognized
        self.on_recognized_callback = None

        # Instantiate the liveness detector
        self.liveness_detector = CombinedLivenessDetector()

        logger.info("✅ FaceRecognitionEngine initialized")

    # ------------------------------------------------------------------
    # Database Loading
    # ------------------------------------------------------------------

    def load_known_faces(self, students):
        """
        Load face encodings from a list of Student ORM objects.
        Call this whenever the DB changes.
        """
        with self.lock:
            self.known_encodings = []
            self.known_names = []
            self.known_ids = []

            for s in students:
                enc = s.get_encoding()
                if enc is not None:
                    self.known_encodings.append(enc)
                    self.known_names.append(s.name)
                    self.known_ids.append(s.student_id)

        logger.info(f"📚 Loaded {len(self.known_encodings)} known face(s)")

    # ------------------------------------------------------------------
    # Single-image Encoding (used during registration)
    # ------------------------------------------------------------------

    def encode_face_from_image(self, image_bytes):
        """
        Given raw image bytes, return the 128-d face encoding.
        Returns (encoding, face_found: bool, message: str)
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, False, "Could not decode image"

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")

        # Illumination compensation fallback for low-light or uneven shadow registration photos
        if len(locations) == 0:
            try:
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                enhanced_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                locs_enhanced = face_recognition.face_locations(enhanced_rgb, model="hog")
                if len(locs_enhanced) > 0:
                    locations = locs_enhanced
                    rgb = enhanced_rgb
            except Exception as e:
                logger.debug(f"CLAHE registration fallback skipped: {e}")

        if len(locations) == 0:
            return None, False, "No face detected in image"
        if len(locations) > 1:
            return None, False, "Multiple faces detected — please use a solo photo"

        encodings = face_recognition.face_encodings(rgb, locations)
        return encodings[0], True, "Face encoded successfully"

    # ------------------------------------------------------------------
    # Multi-face Recognition on a Frame
    # ------------------------------------------------------------------

    def process_frame_ml(self, frame_bgr):
        """
        Runs the heavy ML components (detection, encoding, liveness checking)
        in a separate background thread and updates self.last_recognized.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        small = cv2.resize(frame_bgr, (0, 0),
                            fx=self.scale_factor, fy=self.scale_factor)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ---- Detect ALL faces ----
        face_locs = face_recognition.face_locations(rgb_small, model="hog")

        # Illumination normalization fallback: if no faces detected in poor lighting,
        # apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance channel
        if len(face_locs) == 0:
            try:
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                mean_brightness = gray_small.mean()
                if mean_brightness < 85 or mean_brightness > 195 or gray_small.std() < 35:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    enhanced_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
                    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                    locs_enhanced = face_recognition.face_locations(enhanced_rgb, model="hog")
                    if len(locs_enhanced) > 0:
                        face_locs = locs_enhanced
                        rgb_small = enhanced_rgb
            except Exception as e:
                logger.debug(f"CLAHE detection fallback skipped: {e}")

        # ---- Encode ALL faces ----
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

        scale = int(1.0 / self.scale_factor)

        results = []
        with self.lock:
            for idx, (enc, loc) in enumerate(zip(face_encs, face_locs)):
                name, sid, confidence = self._match_encoding(enc)
                
                is_live = False
                liveness_status = "unknown"  # "unknown", "checking", "live", "spoof"
                liveness_msg = ""

                if sid:
                    top, right, bottom, left = loc
                    top_full = max(0, int(top * scale))
                    right_full = min(frame_bgr.shape[1], int(right * scale))
                    bottom_full = min(frame_bgr.shape[0], int(bottom * scale))
                    left_full = max(0, int(left * scale))

                    face_crop = frame_bgr[top_full:bottom_full, left_full:right_full]

                    is_live, liveness_msg = self.liveness_detector.check(
                        frame_bgr, rgb_small, loc, face_crop, sid
                    )

                    if is_live:
                        liveness_status = "live"
                    elif "Fake detected" in liveness_msg:
                        liveness_status = "spoof"
                    else:
                        liveness_status = "checking"

                results.append((loc, name, sid, confidence, liveness_status, liveness_msg))

                # Trigger attendance callback ONLY if liveness is verified
                if sid and is_live and self._can_mark(sid):
                    self.recent_attendance[sid] = time.time()
                    if self.on_recognized_callback:
                        self.on_recognized_callback(sid, name, confidence)

        self.last_face_locations = face_locs
        self.last_recognized = results
        return results

    def process_frame(self, frame_bgr):
        """
        Process a single BGR frame sequentially.
        Maintained for backwards-compatibility.
        """
        self.frame_count += 1
        recognized_this_frame = []

        # Only run heavy ML every Nth frame
        if self.frame_count % self.process_every_n == 0:
            self.process_frame_ml(frame_bgr)
            recognized_this_frame = self.last_recognized

        # ---- Annotate frame ----
        annotated = self._draw_annotations(frame_bgr, self.last_recognized)
        return annotated, recognized_this_frame

    def _match_encoding(self, face_enc):
        """Compare one encoding against known database."""
        if not self.known_encodings:
            return "Unknown", None, 0.0

        distances = face_recognition.face_distance(self.known_encodings, face_enc)
        best_idx = int(np.argmin(distances))
        best_dist = distances[best_idx]

        if best_dist < self.tolerance:
            # Calibrate confidence relative to tolerance:
            # best_dist = 0.0 -> 1.0 (100%), best_dist = tolerance -> 0.50 (50%)
            confidence = float(np.clip(1.0 - (best_dist / (2.0 * self.tolerance)), 0.50, 1.0))
            return (self.known_names[best_idx],
                    self.known_ids[best_idx],
                    confidence)
        return "Unknown", None, 0.0



    def _can_mark(self, student_id):
        """Check cooldown to avoid duplicate attendance entries."""
        last = self.recent_attendance.get(student_id, 0)
        return (time.time() - last) > self.cooldown_seconds

    def _draw_annotations(self, frame, recognized):
        """Draw bounding boxes and name labels on frame."""
        scale = int(1 / self.scale_factor)

        for item in recognized:
            if len(item) == 6:
                (top, right, bottom, left), name, sid, confidence, liveness_status, liveness_message = item
            elif len(item) == 5:
                (top, right, bottom, left), name, sid, confidence, liveness_status = item
                liveness_message = "Liveness confirmed" if liveness_status == "live" else "Blink to verify"
            else:
                (top, right, bottom, left), name, sid, confidence = item
                liveness_status = "unknown"
                liveness_message = ""

            # Scale back to full resolution
            top    *= scale
            right  *= scale
            bottom *= scale
            left   *= scale

            known = name != "Unknown"
            
            # Select colors based on liveness state
            if known:
                if liveness_status == "live":
                    color  = (0, 200, 0)     # Green
                    color2 = (0, 160, 0)
                    status_lbl = f"[{liveness_message}]"
                elif liveness_status == "spoof":
                    color  = (0, 0, 255)     # Red
                    color2 = (0, 0, 200)
                    status_lbl = f"[{liveness_message}]"
                else: # "checking" or "unknown"
                    color  = (0, 140, 255)   # Amber / Orange
                    color2 = (0, 100, 220)
                    status_lbl = f"[{liveness_message}]"
            else:
                color  = (0, 60, 220)        # Reddish-blue for unknown
                color2 = (0, 40, 180)
                status_lbl = ""

            # Rounded rectangle (OpenCV doesn't have it, simulate with lines)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Label background
            label = f"{name} {confidence*100:.0f}% {status_lbl}".strip() if known else "Unknown"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(frame, (left, bottom - th - 10),
                          (left + tw + 8, bottom), color2, -1)
            cv2.putText(frame, label, (left + 4, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        # Frame counter overlay
        cv2.putText(frame, f"Faces: {len(recognized)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
        return frame

    # ------------------------------------------------------------------
    # Frame → base64 JPEG (for streaming to browser)
    # ------------------------------------------------------------------

    @staticmethod
    def frame_to_base64(frame_bgr):
        _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode('utf-8')


# ------------------------------------------------------------------
# Camera Manager — runs capture loop in background thread
# ------------------------------------------------------------------

class CameraManager:
    """Manages webcam capture + recognition loop with multi-threaded performance tuning."""

    def __init__(self, engine: FaceRecognitionEngine, camera_index=0, processing_fps=10):
        self.engine = engine
        self.camera_index = camera_index
        self.processing_fps = processing_fps
        self.cap = None
        self.running = False
        
        self.capture_thread = None
        self.processing_thread = None
        self.stream_thread = None
        
        self.latest_raw_frame = None
        self.raw_frame_lock = threading.Lock()
        
        self.latest_frame_b64 = None
        self.frame_lock = threading.Lock()
        
        self.new_frame_available = False
        self.fps = 0

    def start(self):
        if self.running:
            return False
        # Use DirectShow backend on Windows for faster initialization and lower latency
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"❌ Cannot open camera at index {self.camera_index}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        
        self.capture_thread.start()
        self.processing_thread.start()
        self.stream_thread.start()
        
        logger.info("📷 Camera started with decoupled parallel threads")
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("📷 Camera stopped")

    def _reconnect_camera(self):
        """Attempts to re-open the camera if frame drops or disconnections occur."""
        try:
            if self.cap:
                self.cap.release()
            time.sleep(0.2)
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                logger.info("✅ Camera successfully reconnected")
        except Exception as e:
            logger.debug(f"Camera reconnect attempt skipped: {e}")

    def _capture_loop(self):
        consecutive_drops = 0
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self._reconnect_camera()
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_drops += 1
                if consecutive_drops >= 30:  # ~1s continuous failure
                    logger.warning("⚠️ Camera frame drops detected, attempting auto-reconnect...")
                    self._reconnect_camera()
                    consecutive_drops = 0
                time.sleep(0.02)
                continue

            consecutive_drops = 0
            with self.raw_frame_lock:
                self.latest_raw_frame = frame
                self.new_frame_available = True
            time.sleep(0.01)

    def _processing_loop(self):
        interval = 1.0 / self.processing_fps
        while self.running:
            t0 = time.time()
            frame_to_process = None
            with self.raw_frame_lock:
                if self.new_frame_available and self.latest_raw_frame is not None:
                    frame_to_process = self.latest_raw_frame.copy()
                    self.new_frame_available = False
            
            if frame_to_process is not None:
                # Run background ML recognition and liveness
                self.engine.process_frame_ml(frame_to_process)
            
            elapsed = time.time() - t0
            sleep_time = max(0.005, interval - elapsed)
            time.sleep(sleep_time)

    def _stream_loop(self):
        t0 = time.time()
        fc = 0
        while self.running:
            frame_to_stream = None
            with self.raw_frame_lock:
                if self.latest_raw_frame is not None:
                    frame_to_stream = self.latest_raw_frame.copy()
            
            if frame_to_stream is not None:
                # Draw annotations on raw frame using last known results
                annotated = self.engine._draw_annotations(frame_to_stream, self.engine.last_recognized)
                b64 = FaceRecognitionEngine.frame_to_base64(annotated)
                
                with self.frame_lock:
                    self.latest_frame_b64 = b64
                
                fc += 1
                elapsed = time.time() - t0
                if elapsed >= 1.0:
                    self.fps = fc / elapsed
                    fc = 0
                    t0 = time.time()
            
            time.sleep(0.033)

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame_b64
