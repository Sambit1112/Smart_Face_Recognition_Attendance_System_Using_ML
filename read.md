# Smart Face Attendance System — Reference Guide

This document serves as a developer reference guide detailing the system's machine learning architectures, file structure, and capability to handle simultaneous multi-face attendance capture.

---

## 👥 1. Simultaneous Multi-Face Capture

The system supports detecting, verifying, and logging attendance for **multiple faces simultaneously** in real time from the webcam feed.

### How It Works:
1. **Detection & Embedding Extraction**: In [process_frame_ml](file:///d:/Face%20Attendance/models/face_engine.py#L102), the system runs `face_recognition.face_locations` and `face_recognition.face_encodings` over the downscaled frame. This processes all visible faces in parallel rather than sequentially.
2. **Concurrent Liveness Checking**: The [CombinedLivenessDetector](file:///d:/Face%20Attendance/models/liveness.py#L145) runs checks on each face:
   - **Texture Analysis**: Instant sharpness check using Laplacian variance and screen reflection check in the HSV color space.
   - **Blink Tracking**: The [BlinkLivenessDetector](file:///d:/Face%20Attendance/models/liveness.py#L10) manages separate blink state dictionaries keyed by `student_id` in [check_frame](file:///d:/Face%20Attendance/models/liveness.py#L23). Multiple students can perform blink verifications at the same time.
3. **Independent Marking & Cooldowns**: Once a student's face passes all checks, their attendance is logged via the callback. A 30-second cooldown is enforced individually per student to avoid duplicate logs.

> [!NOTE]
> During **registration** ([encode_face_from_image](file:///d:/Face%20Attendance/models/face_engine.py#L77)), multiple faces are rejected to ensure that only a clean, solo photo is stored in the database for each student.

---

## 🧠 2. Machine Learning Model Architecture

The system uses a hybrid pipeline combining deep learning (via the wrapper library `face_recognition` / `dlib`) and statistical computer vision checks (via `OpenCV`).

| Model / Algorithmic Check | Primary Task | File & Method Reference | Description & Mechanics |
| :--- | :--- | :--- | :--- |
| **HOG + Linear SVM** | Face Detection | [process_frame_ml](file:///d:/Face%20Attendance/models/face_engine.py#L112) | Finds face coordinates (bounding boxes) in a frame. Optimized by running on a downscaled $0.25\times$ resolution frame. |
| **ResNet-34** (Deep Metric Learning) | Feature Extraction & Recognition | [process_frame_ml](file:///d:/Face%20Attendance/models/face_engine.py#L115) | Generates a 128-dimensional vector representing the face. Compares it against database encodings using L2 / Euclidean distance with a tolerance threshold of `0.50`. |
| **68-Point Face Landmark Predictor** | Alignment & Blink Tracking | [BlinkLivenessDetector.check_frame](file:///d:/Face%20Attendance/models/liveness.py#L55) | Extracts keypoints mapping the brows, jaw, nose, and eyes. Computes the **Eye Aspect Ratio (EAR)** using vertical-to-horizontal eye distances to verify physical blinks. |
| **Laplacian Filter** | Print Photo Anti-Spoofing | [TextureLivenessDetector.is_live](file:///d:/Face%20Attendance/models/liveness.py#L110) | Convolves the face crop with a Laplacian kernel. If the edge-sharpness variance is below `80.0`, the system flag-checks the input as flat printed paper (spoof). |
| **HSV Glare Masking** | Digital Screen Anti-Spoofing | [TextureLivenessDetector.is_live](file:///d:/Face%20Attendance/models/liveness.py#L126) | Converts the crop to HSV and flags high value ($V > 240$) and low saturation ($S < 20$) regions to catch reflections from phones/tablets. |

---

## 📂 3. Key Files & Components Reference

* **Main Server & Endpoints**: [backend/app.py](file:///d:/Face%20Attendance/backend/app.py)
  - Manages WebSockets streaming (`Socket.IO`), Flask API routes, and handles attendance events.
* **ML Recognition Core**: [models/face_engine.py](file:///d:/Face%20Attendance/models/face_engine.py)
  - Houses the [FaceRecognitionEngine](file:///d:/Face%20Attendance/models/face_engine.py#L16) and [CameraManager](file:///d:/Face%20Attendance/models/face_engine.py#L291) which decoupled webcam frame acquisition and processing onto separate threads.
* **Liveness Detection & Spoof Mitigation**: [models/liveness.py](file:///d:/Face%20Attendance/models/liveness.py)
  - Features the texture and blink checking modules used to verify physical presence.
* **Database Schema & ORM**: [database/models.py](file:///d:/Face%20Attendance/database/models.py)
  - Contains SQLAlchemy database structures storing student profiles, their serialized 128-D face encodings, and attendance logs.
* **Registration & Database Onboarding Operations**: [utils/registration.py](file:///d:/Face%20Attendance/utils/registration.py)
  - Handles uploading photos, extracting encodings, updating the DB, and refreshing the recognition engine memory cache.
* **Attendance Business Logic**: [utils/attendance_manager.py](file:///d:/Face%20Attendance/utils/attendance_manager.py)
  - Manages the daily attendance registers, checking check-in/check-out limits, calculating statistics, and handling CSV data exports.
