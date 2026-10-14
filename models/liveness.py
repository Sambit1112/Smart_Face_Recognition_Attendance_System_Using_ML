import os
import time
import logging
import collections
import numpy as np
import cv2
import urllib.request

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  1. FRAME QUALITY GATE
# --------------------------------------------------------------------------- #
class FrameQualityGate:
    """
    Rejects poor-quality frames BEFORE they reach the liveness model.
    Most false rejections in real deployments come from bad input, not a bad
    model, so this gate improves effective accuracy for free.
    """

    def __init__(self, min_face_size=30, blur_threshold=4.0,
                 min_brightness=15, max_brightness=245):
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def check(self, face_crop_bgr):
        """
        Returns (passed: bool, reason: str)
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return False, "no_face_image"
            
        h, w = face_crop_bgr.shape[:2]
        if min(h, w) < self.min_face_size:
            return False, "face_too_small"

        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

        # Blur check — Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.blur_threshold:
            return False, "too_blurry"

        # Exposure check
        mean_brightness = gray.mean()
        if mean_brightness < self.min_brightness:
            return False, "too_dark"
        if mean_brightness > self.max_brightness:
            return False, "overexposed"

        return True, "ok"


# --------------------------------------------------------------------------- #
#  2. PASSIVE CNN LIVENESS (PRIMARY LAYER)
# --------------------------------------------------------------------------- #
class PassiveCNNLiveness:
    """
    Wraps a pretrained passive anti-spoofing CNN (e.g. MiniFASNet / DeepPixBiS
    exported to ONNX) via OpenCV's DNN module — no PyTorch/TF runtime needed
    at inference time, which keeps the Flask server lightweight.

    Model setup:
      1. Train/obtain a binary (real=1 / spoof=0) face anti-spoofing model.
         Recommended starting point: minivision-ai/Silent-Face-Anti-Spoofing
         (MiniFASNetV2), converted to ONNX.
      2. Place the .onnx file at `model_path` below.
      3. If no model file is found, this class automatically falls back to a
         lightweight LBP-texture heuristic so the pipeline still runs
         end-to-end during development (NOT for production accuracy).
    """

    def __init__(self, model_path=None, input_size=80):
        self.input_size = input_size
        self.net = None
        self.mode = "fallback_texture"
        
        # Resolve model path dynamically in same directory as this module
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            self.model_path = os.path.join(model_dir, "MiniFASNetV2.onnx")
        else:
            self.model_path = model_path
            
        self.model_url = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"
        
        self.load_model()

    def load_model(self):
        try:
            # Proactively download model weights if missing
            if not os.path.exists(self.model_path) and "MiniFASNetV2.onnx" in self.model_path:
                logger.info(f"[PassiveCNNLiveness] Downloading model weights from {self.model_url}...")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                req = urllib.request.Request(
                    self.model_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, timeout=30) as response, open(self.model_path, 'wb') as out_file:
                    out_file.write(response.read())
                logger.info("[PassiveCNNLiveness] Model weights downloaded successfully.")
            
            if os.path.exists(self.model_path):
                logger.info(f"[PassiveCNNLiveness] Loading Silent-Face-Anti-Spoofing model from {self.model_path}...")
                self.net = cv2.dnn.readNet(self.model_path)
                self.mode = "cnn"
                logger.info("[PassiveCNNLiveness] Silent-Face-Anti-Spoofing model loaded successfully.")
            else:
                logger.warning(f"[PassiveCNNLiveness] No model found at '{self.model_path}'. Using fallback_texture.")
        except Exception as e:
            logger.error(f"[PassiveCNNLiveness] Failed to load/download ONNX model: {e}")
            logger.warning("[PassiveCNNLiveness] Falling back to texture heuristic.")
            self.net = None
            self.mode = "fallback_texture"

    # ---- primary path: real CNN inference -------------------------------- #
    def _predict_cnn(self, face_crop_bgr):
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return 0.0
            
        # Convert to float32 (MiniFASNet ONNX expects raw float pixels in [0, 255])
        img_float = face_crop_bgr.astype(np.float32)
        
        # Convert to NCHW blob (retains BGR order, scales by 1.0, resizes to 80x80)
        blob = cv2.dnn.blobFromImage(
            img_float,
            scalefactor=1.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()
        out = out.flatten()

        # MiniFASNetV2 outputs 3 classes: [spoof1, real, spoof2]
        if out.size == 3:
            # Softmax calculation
            exp_outputs = np.exp(out - np.max(out))
            probs = exp_outputs / np.sum(exp_outputs)
            real_prob = float(probs[1])
        elif out.size >= 2:
            real_prob = float(out[1])
        else:
            real_prob = float(1 / (1 + np.exp(-out[0])))  # sigmoid fallback
        return real_prob

    # ---- fallback path: LBP micro-texture heuristic ----------------------- #
    def _predict_texture_fallback(self, face_crop_bgr):
        """
        Cheap, no-training heuristic used only when no CNN model is present.
        Real skin has fine, irregular micro-texture; printed photos and
        screens tend to be smoother or show regular print/pixel patterns.
        This is a rough proxy, not a replacement for a trained model.
        """
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # Local Binary Pattern
        lbp = self._local_binary_pattern(gray)
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59), density=True)

        # Higher texture entropy tends to correlate with real skin;
        # very peaked histograms (low entropy) correlate with flat/print spoof.
        entropy = -np.sum(hist * np.log2(hist + 1e-9))
        max_entropy = np.log2(59)
        score = float(np.clip(entropy / max_entropy, 0.0, 1.0))
        return score

    @staticmethod
    def _local_binary_pattern(gray_img):
        h, w = gray_img.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        center = gray_img[1:-1, 1:-1]
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            neighbour = gray_img[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
            lbp |= ((neighbour >= center).astype(np.uint8) << bit)
        return lbp

    def predict(self, face_crop_bgr):
        """
        Returns real_prob in [0, 1] — probability the face is a live/real face.
        """
        if self.mode == "cnn":
            return self._predict_cnn(face_crop_bgr)
        return self._predict_texture_fallback(face_crop_bgr)


# --------------------------------------------------------------------------- #
#  3. FREQUENCY-DOMAIN TEXTURE ANALYSIS (SECONDARY / LIGHTWEIGHT FILTER)
# --------------------------------------------------------------------------- #
class FrequencyDomainAnalyzer:
    """
    Detects screen-replay attacks (phone/tablet showing a face) via moire
    interference patterns, which show up as distinct high-frequency peaks
    in the 2D FFT spectrum that a real face / printed photo does not produce.

    This is a lightweight, training-free secondary filter that catches the
    specific spoof type the primary CNN sometimes misses.
    """

    def __init__(self, high_freq_ratio_threshold=0.035):
        self.threshold = high_freq_ratio_threshold

    def analyze(self, face_crop_bgr):
        """
        Returns real_prob in [0, 1] — higher means fewer moire artifacts
        (more likely real / less likely a screen replay).
        """
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)

        # 2D FFT + shift zero-frequency to center
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Mask out the low-frequency core (general image content)
        low_freq_radius = 20
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        low_mask = dist_from_center <= low_freq_radius
        high_mask = ~low_mask

        total_energy = magnitude.sum() + 1e-9
        high_freq_energy = magnitude[high_mask].sum()
        high_freq_ratio = high_freq_energy / total_energy

        # Screen replays show elevated, structured high-frequency energy
        # (moire) relative to real faces. Convert ratio to a pseudo-probability:
        # For real faces, high_freq_ratio is typically <= 0.65.
        if high_freq_ratio <= 0.65:
            real_prob = 1.0
        else:
            real_prob = float(np.clip(1.0 - (high_freq_ratio - 0.65) / 0.20, 0.0, 1.0))
        return real_prob


# --------------------------------------------------------------------------- #
#  4. MULTI-FRAME TEMPORAL BUFFER (reduces single-frame noise)
# --------------------------------------------------------------------------- #
class TemporalBuffer:
    """
    Accumulates liveness scores for a tracked face across a short burst of
    frames (~1 second at walk-up) and returns an averaged, more robust score
    once enough frames are collected.
    """

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffers = collections.defaultdict(
            lambda: collections.deque(maxlen=window_size)
        )

    def add_score(self, track_id, cnn_score, freq_score):
        now = time.time()
        if track_id in self.buffers:
            # Clean up stale entries (> 5.0 seconds old)
            filtered = [e for e in self.buffers[track_id] if now - e[2] <= 5.0]
            self.buffers[track_id] = collections.deque(filtered, maxlen=self.window_size)
        self.buffers[track_id].append((cnn_score, freq_score, now))

    def is_ready(self, track_id):
        return len(self.buffers[track_id]) >= self.window_size

    def get_averaged_scores(self, track_id):
        entries = self.buffers[track_id]
        cnn_avg = float(np.mean([e[0] for e in entries]))
        freq_avg = float(np.mean([e[1] for e in entries]))
        return cnn_avg, freq_avg

    def reset(self, track_id):
        self.buffers.pop(track_id, None)


# --------------------------------------------------------------------------- #
#  5. FUSION LAYER
# --------------------------------------------------------------------------- #
class LivenessFusion:
    """
    Combines primary CNN score and secondary frequency-domain score into a
    final decision. Weighted average by default; swap in a trained logistic
    regression / small MLP here once you have labelled fusion data from your
    own deployment for a further accuracy bump.
    """

    def __init__(self, cnn_weight=0.7, freq_weight=0.3, accept_threshold=0.6):
        assert abs(cnn_weight + freq_weight - 1.0) < 1e-6
        self.cnn_weight = cnn_weight
        self.freq_weight = freq_weight
        self.accept_threshold = accept_threshold

    def fuse(self, cnn_score, freq_score):
        final_score = self.cnn_weight * cnn_score + self.freq_weight * freq_score
        is_live = final_score >= self.accept_threshold
        return is_live, final_score


# --------------------------------------------------------------------------- #
#  6. END-TO-END PIPELINE
# --------------------------------------------------------------------------- #
class LivenessPipeline:
    """
    Convenience wrapper tying all stages together. Use this directly from
    your Flask route / attendance-marking loop.
    """

    def __init__(self, model_path=None,
                 use_temporal=True, window_size=3,
                 accept_threshold=0.6):
        self.quality_gate = FrameQualityGate()
        self.cnn_model = PassiveCNNLiveness(model_path=model_path)
        self.freq_analyzer = FrequencyDomainAnalyzer()
        self.fusion = LivenessFusion(accept_threshold=accept_threshold)
        self.use_temporal = use_temporal
        self.temporal_buffer = TemporalBuffer(window_size=window_size) if use_temporal else None

    def evaluate_single_frame(self, face_crop_bgr):
        """
        Single-frame liveness check. Returns a dict with the decision and
        component scores — useful for logging / debugging / report metrics.
        """
        passed, reason = self.quality_gate.check(face_crop_bgr)
        if not passed:
            return {
                "accepted": False,
                "stage_failed": "quality_gate",
                "reason": reason,
                "cnn_score": None,
                "freq_score": None,
                "final_score": None,
            }

        cnn_score = self.cnn_model.predict(face_crop_bgr)
        freq_score = self.freq_analyzer.analyze(face_crop_bgr)
        is_live, final_score = self.fusion.fuse(cnn_score, freq_score)

        return {
            "accepted": is_live,
            "stage_failed": None if is_live else "liveness_fusion",
            "reason": "live" if is_live else "spoof_suspected",
            "cnn_score": round(cnn_score, 4),
            "freq_score": round(freq_score, 4),
            "final_score": round(final_score, 4),
        }

    def evaluate_tracked_face(self, track_id, face_crop_bgr):
        """
        Multi-frame version — call this once per frame for a tracked face
        (e.g. using a simple centroid tracker keyed by track_id) during the
        ~1s walk-up window. Returns None until enough frames are collected,
        then returns the same dict shape as evaluate_single_frame.
        """
        if not self.use_temporal:
            return self.evaluate_single_frame(face_crop_bgr)

        passed, reason = self.quality_gate.check(face_crop_bgr)
        if not passed:
            return {
                "accepted": False,
                "stage_failed": "quality_gate",
                "reason": reason,
                "cnn_score": None,
                "freq_score": None,
                "final_score": None,
            }

        cnn_score = self.cnn_model.predict(face_crop_bgr)
        freq_score = self.freq_analyzer.analyze(face_crop_bgr)
        self.temporal_buffer.add_score(track_id, cnn_score, freq_score)

        if not self.temporal_buffer.is_ready(track_id):
            return None  # keep collecting frames

        cnn_avg, freq_avg = self.temporal_buffer.get_averaged_scores(track_id)
        is_live, final_score = self.fusion.fuse(cnn_avg, freq_avg)
        self.temporal_buffer.reset(track_id)

        return {
            "accepted": is_live,
            "stage_failed": None if is_live else "liveness_fusion",
            "reason": "live" if is_live else "spoof_suspected",
            "cnn_score": round(cnn_avg, 4),
            "freq_score": round(freq_avg, 4),
            "final_score": round(final_score, 4),
        }


class TextureSettingsCompat:
    """
    Compatibility layer mapping app.py settings requests to FrameQualityGate.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._glare_threshold = 0.02

    @property
    def laplacian_threshold(self):
        return self.pipeline.quality_gate.blur_threshold

    @laplacian_threshold.setter
    def laplacian_threshold(self, val):
        self.pipeline.quality_gate.blur_threshold = val

    @property
    def glare_threshold(self):
        return self._glare_threshold

    @glare_threshold.setter
    def glare_threshold(self, val):
        self._glare_threshold = val


# --------------------------------------------------------------------------- #
#  7. ENGINE COMPATIBILITY / ADAPTER LAYER
# --------------------------------------------------------------------------- #
class CombinedLivenessDetector:
    """
    Adapter class to plug the new LivenessPipeline into the existing face recognition engine.
    Performs frame coordinates scaling and executes liveness checks.
    """
    def __init__(self):
        # Resolve path dynamically inside same directory as this module
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(model_dir, "MiniFASNetV2.onnx")
        
        # Instantiate pipeline with temporal tracking enabled
        self.pipeline = LivenessPipeline(model_path=self.model_path, use_temporal=True, window_size=3, accept_threshold=0.55)
        
        # Expose texture attribute for settings compatibility with app.py
        self.texture = TextureSettingsCompat(self.pipeline)
        
        logger.info(f"CombinedLivenessDetector initialized with pipeline mode: {self.pipeline.cnn_model.mode}")

    def check(self, frame_bgr, rgb_small, face_rect_small, face_crop, student_id):
        """
        Perform liveness check using the new LivenessPipeline.
        Returns: (is_live: bool, message: str)
        """
        # 1. Quality gate check on tight crop
        quality_passed, quality_reason = self.pipeline.quality_gate.check(face_crop)
        if not quality_passed:
            logger.warning(f"⚠️ Quality Gate rejected face for {student_id}: {quality_reason}")
            return False, f"Quality failed: {quality_reason}"

        # Determine scaling factors dynamically
        h_full, w_full = frame_bgr.shape[:2]
        h_small, w_small = rgb_small.shape[:2]
        scale_x = w_full / w_small
        scale_y = h_full / h_small
        
        top, right, bottom, left = face_rect_small
        top_full = int(top * scale_y)
        right_full = int(right * scale_x)
        bottom_full = int(bottom * scale_y)
        left_full = int(left * scale_x)
        face_rect_full = (top_full, right_full, bottom_full, left_full)

        # 2. Extract 2.7x margin crop specifically for the CNN liveness model
        cnn_face_crop = self.crop_face_with_margin(frame_bgr, face_rect_full)

        # 3. Predict frame scores
        cnn_score = self.pipeline.cnn_model.predict(cnn_face_crop)
        freq_score = self.pipeline.freq_analyzer.analyze(face_crop)

        # 4. Process temporal averaging
        if self.pipeline.use_temporal:
            self.pipeline.temporal_buffer.add_score(student_id, cnn_score, freq_score)
            
            if not self.pipeline.temporal_buffer.is_ready(student_id):
                # Still collecting frames for student tracking window, return checking state
                return False, "Checking liveness..."
                
            # Averaged scores once buffer is full
            cnn_score, freq_score = self.pipeline.temporal_buffer.get_averaged_scores(student_id)
            self.pipeline.temporal_buffer.reset(student_id)

        # 5. Fuse scores
        is_live, final_score = self.pipeline.fusion.fuse(cnn_score, freq_score)
        
        cnn_mode = self.pipeline.cnn_model.mode
        logger.info(f"Liveness decision for {student_id} -> Final: {final_score:.4f} (CNN [{cnn_mode}]: {cnn_score:.4f}, Freq: {freq_score:.4f})")

        if not is_live:
            return False, f"Fake detected (Score: {final_score:.2f})"

        return True, "Liveness confirmed"

    def crop_face_with_margin(self, frame_bgr, face_rect_full, scale_factor=2.7):
        top, right, bottom, left = face_rect_full
        w = right - left
        h = bottom - top
        
        cx = left + w // 2
        cy = top + h // 2
        
        size = int(max(w, h) * scale_factor)
        
        # Calculate raw crop coordinates (may be out of bounds)
        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = x1 + size
        y2 = y1 + size
        
        # Calculate padding needed for each side
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - frame_bgr.shape[1])
        pad_bottom = max(0, y2 - frame_bgr.shape[0])
        
        # Crop bounds clamped to frame
        x1_clamp = max(0, x1)
        y1_clamp = max(0, y1)
        x2_clamp = min(frame_bgr.shape[1], x2)
        y2_clamp = min(frame_bgr.shape[0], y2)
        
        crop = frame_bgr[y1_clamp:y2_clamp, x1_clamp:x2_clamp]
        
        # Apply black padding if out of bounds to keep it square and centered
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            
        return crop
