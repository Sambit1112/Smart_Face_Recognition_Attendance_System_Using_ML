// ── Config ─────────────────────────────────────────────────────────────────
const API = '';  // Same-origin; change to http://localhost:5000 if separate
let socket;
let cameraRunning = false;
let liveLogItems = [];

let regStream = null;
let regPhotoMode = 'webcam'; // 'webcam' or 'upload'
let allRegisteredStudents = []; // Cached list of all students
let currentFilteredBatch = 'All'; // Currently active batch filter

// ── Global Fetch Interceptor ────────────────────────────────────────────────
const originalFetch = window.fetch;
window.fetch = async function(...args) {
  const response = await originalFetch(...args);
  if (response.status === 401) {
    const url = args[0];
    if (typeof url === 'string' && !url.includes('/api/auth/status') && !url.includes('/api/auth/login')) {
      showLoginScreen();
      toast('🔒', 'Session expired. Please log in again.', 'error');
    }
  }
  return response;
};

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  startClock();
  setTodayDate();
  checkAuthStatus();

  // Set date filter to today
  const today = new Date().toISOString().split('T')[0];
  document.getElementById('dateFilter').value = today;
});

// ── Clock ───────────────────────────────────────────────────────────────────
function startClock() {
  const el = document.getElementById('clockDisplay');
  setInterval(() => {
    el.textContent = new Date().toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  }, 1000);
}

function setTodayDate() {
  const el = document.getElementById('todayDate');
  if (el) el.textContent = new Date().toLocaleDateString('en-US', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
  });
}

// ── Socket ─────────────────────────────────────────────────────────────────
function connectSocket() {
  try {
    socket = io(window.location.origin);

    socket.on('connect', () => console.log('Socket connected'));

    socket.on('video_frame', data => {
      const img = document.getElementById('videoFeed');
      if (img) img.src = 'data:image/jpeg;base64,' + data.frame;
      const fpsBadge = document.getElementById('fpsBadge');
      if (fpsBadge) fpsBadge.textContent = `FPS: ${data.fps}`;
    });

    socket.on('attendance_update', data => {
      if (data.action === 'rejected') {
        toast('⚠️', data.message || 'Attendance rejected', 'error');
        return;
      }
      const rec = data.record;
      addLiveLogItem(rec);
      updateStats(data.stats);
      if (document.getElementById('page-attendance').classList.contains('active')) {
        loadTodayAttendance();
      }
      toast(
        '✅',
        `${rec.student_name} — checked in at ${new Date().toLocaleTimeString()}`,
        'success'
      );
    });

    socket.on('stats_update', stats => updateStats(stats));

  } catch(e) {
    console.warn('Socket not available:', e);
  }
}

// ── Navigation ──────────────────────────────────────────────────────────────
function showPage(name) {
  // Always stop the registration camera when switching pages
  stopRegCamera();

  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => {
    if (n.getAttribute('onclick')?.includes(name)) n.classList.add('active');
  });

  // Load data for the page
  if (name === 'attendance') loadTodayAttendance();
  if (name === 'students') loadStudents();
  if (name === 'dashboard') refreshAll();
  if (name === 'settings') loadSettings();

  // Automatically start registration camera if navigating to register
  if (name === 'register') {
    if (cameraRunning) {
      stopBackendCameraAndStartReg();
    } else {
      switchPhotoMode('webcam');
    }
  }
}

async function stopBackendCameraAndStartReg() {
  try {
    const r = await fetch(API + '/api/camera/stop', { method: 'POST' });
    const d = await r.json();
    if (d.success) {
      cameraRunning = false;
      updateCameraUI();
      toast('⏹', 'System camera stopped to release webcam hardware lock.', '');
    }
  } catch(e) {}
  switchPhotoMode('webcam');
}

// ── Camera ──────────────────────────────────────────────────────────────────
async function toggleCamera() {
  const endpoint = cameraRunning ? '/api/camera/stop' : '/api/camera/start';
  try {
    const r = await fetch(API + endpoint, { method: 'POST' });
    const d = await r.json();
    if (d.success) {
      cameraRunning = !cameraRunning;
      updateCameraUI();
      toast(cameraRunning ? '📷' : '⏹', d.message, cameraRunning ? 'success' : '');
    } else {
      toast('⚠️', d.message, 'error');
    }
  } catch(e) {
    toast('❌', 'Cannot connect to server. Is the backend running?', 'error');
  }
}

function updateCameraUI() {
  const dot = document.getElementById('camDot');
  const btn = document.getElementById('camToggleBtn');
  const btn2 = document.getElementById('camBtn2');
  const feed = document.getElementById('videoFeed');
  const placeholder = document.getElementById('videoPlaceholder');
  const overlay = document.getElementById('videoOverlay');

  if (cameraRunning) {
    dot?.classList.add('active');
    if (btn) btn.textContent = '⏹ Stop Camera';
    if (btn2) btn2.textContent = '⏹ Stop';
    if (feed) feed.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
    if (overlay) overlay.style.display = 'flex';
  } else {
    dot?.classList.remove('active');
    if (btn) btn.textContent = '▶ Start Camera';
    if (btn2) btn2.textContent = '▶ Start';
    if (feed) feed.style.display = 'none';
    if (placeholder) placeholder.style.display = 'flex';
    if (overlay) overlay.style.display = 'none';
  }
}

// ── Registration Camera Controls ───────────────────────────────────────────
function switchPhotoMode(mode) {
  regPhotoMode = mode;
  document.querySelectorAll('.photo-tab').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.photo-mode-container').forEach(c => c.classList.remove('active'));

  if (mode === 'webcam') {
    document.getElementById('tabWebcam').classList.add('active');
    document.getElementById('webcamContainer').classList.add('active');
    selectedPhoto = null;
    startRegCamera();
  } else {
    document.getElementById('tabUpload').classList.add('active');
    document.getElementById('uploadContainer').classList.add('active');
    stopRegCamera();
    selectedPhoto = null;
    document.getElementById('regPhotoPreview').style.display = 'none';
    document.getElementById('regPhotoPreview').src = '';
    document.getElementById('regVideo').style.display = 'none';
    document.getElementById('webcamPlaceholder').style.display = 'flex';
    document.getElementById('regCaptureBtn').style.display = 'none';
    document.getElementById('regRetakeBtn').style.display = 'none';
  }
}

async function startRegCamera() {
  const video = document.getElementById('regVideo');
  const placeholder = document.getElementById('webcamPlaceholder');
  const captureBtn = document.getElementById('regCaptureBtn');
  const retakeBtn = document.getElementById('regRetakeBtn');
  const preview = document.getElementById('regPhotoPreview');

  try {
    stopRegCamera(); // Clean start
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("InsecureContext");
    }
    
    regStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });
    video.srcObject = regStream;
    video.style.display = 'block';
    placeholder.style.display = 'none';
    preview.style.display = 'none';
    preview.src = '';
    captureBtn.style.display = 'inline-block';
    retakeBtn.style.display = 'none';
    selectedPhoto = null;
  } catch (e) {
    console.warn('Webcam permission denied or unavailable:', e);
    if (e.message === "InsecureContext" || window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      toast('🔒', 'Webcam requires a Secure Context. Please use http://localhost:5000 or configure chrome://flags/#unsafely-treat-insecure-origin-as-secure', 'error');
    } else {
      toast('⚠️', 'Webcam locked or unavailable. Ensure main camera is stopped.', 'error');
    }
  }
}

function stopRegCamera() {
  if (regStream) {
    regStream.getTracks().forEach(track => track.stop());
    regStream = null;
  }
}

function captureRegPhoto() {
  const video = document.getElementById('regVideo');
  const preview = document.getElementById('regPhotoPreview');
  const captureBtn = document.getElementById('regCaptureBtn');
  const retakeBtn = document.getElementById('regRetakeBtn');

  if (!regStream) return;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    selectedPhoto = new File([blob], "student_capture.jpg", { type: "image/jpeg" });
    preview.src = URL.createObjectURL(blob);
    preview.style.display = 'block';
    video.style.display = 'none';
    captureBtn.style.display = 'none';
    retakeBtn.style.display = 'inline-block';
    stopRegCamera();
  }, 'image/jpeg', 0.9);
}

function retakeRegPhoto() {
  startRegCamera();
}


// ── Stats & Dashboard ───────────────────────────────────────────────────────
async function refreshAll() {
  await Promise.all([loadStats(), loadWeekly(), loadTodayAttendance()]);
}

async function loadStats() {
  try {
    const r = await fetch(API + '/api/attendance/stats');
    const d = await r.json();
    if (d.success) updateStats(d.stats);
  } catch(e) {}
}

function updateStats(stats) {
  setText('statTotal', stats.total_students ?? '—');
  setText('statPresent', stats.present_today ?? '—');
  setText('statAbsent', stats.absent_today ?? '—');
  setText('statRate', stats.attendance_rate != null ? stats.attendance_rate + '%' : '—');
  setText('liveCount', (stats.present_today ?? 0) + ' today');
}

async function loadWeekly() {
  try {
    const r = await fetch(API + '/api/attendance/weekly');
    const d = await r.json();
    if (d.success) renderWeeklyChart(d.trend);
  } catch(e) {}
}

function renderWeeklyChart(trend) {
  const container = document.getElementById('weeklyChart');
  if (!container) return;
  const max = Math.max(...trend.map(t => t.count), 1);
  container.innerHTML = trend.map(t => {
    const pct = Math.max((t.count / max) * 100, 4);
    const label = new Date(t.date).toLocaleDateString('en', { weekday: 'short' });
    return `
      <div class="chart-bar-wrap">
        <div class="chart-bar" style="height:${pct}%" data-val="${t.count}"></div>
        <div class="chart-label">${label}</div>
      </div>`;
  }).join('');
}

// ── Attendance Table ─────────────────────────────────────────────────────────
async function loadTodayAttendance() {
  const dateVal = document.getElementById('dateFilter')?.value;
  const today = new Date().toISOString().split('T')[0];
  const target = dateVal || today;
  await loadAttendanceForDate(target);
}

async function loadAttendanceByDate() {
  const dateVal = document.getElementById('dateFilter').value;
  if (dateVal) await loadAttendanceForDate(dateVal);
}

async function loadAttendanceForDate(dateStr) {
  try {
    const r = await fetch(API + `/api/attendance/date/${dateStr}`);
    const d = await r.json();
    renderAttendanceTable(d.records || []);
  } catch(e) {}
}

function renderAttendanceTable(records) {
  const tbody = document.getElementById('attendanceTable');
  if (!tbody) return;
  if (!records.length) {
    tbody.innerHTML = '<tr><td colspan="6"><div class="empty-state"><div class="empty-icon">📋</div>No records for this date</div></td></tr>';
    return;
  }
  tbody.innerHTML = records.map(r => `
    <tr>
      <td><strong>${r.student_name}</strong></td>
      <td><span class="mono">${r.student_id}</span></td>
      <td>${r.session || '—'}</td>
      <td><span class="mono">${fmtTime(r.check_in_time)}</span></td>
      <td>
        <div class="conf-bar">
          <div class="conf-track"><div class="conf-fill" style="width:${r.confidence}%"></div></div>
          <span style="font-size:11px;color:var(--muted);font-family:var(--mono)">${r.confidence}%</span>
        </div>
      </td>
      <td><span class="badge ${r.status.toLowerCase()}">${r.status}</span></td>
    </tr>`).join('');
}

// ── Students Table ───────────────────────────────────────────────────────────
async function loadStudents() {
  try {
    const r = await fetch(API + '/api/students');
    const d = await r.json();
    const count = document.getElementById('studentCount');
    if (count) count.textContent = `${d.total} student(s) registered`;
    allRegisteredStudents = d.students || [];
    renderStudentsTable(allRegisteredStudents);
  } catch(e) {}
}

function renderStudentsTable(students) {
  const tbody = document.getElementById('studentsTable');
  if (!tbody) return;

  // Dynamically render batch selection tabs
  renderBatchTabs(students);

  // Filter students based on currently selected batch tab
  const filtered = currentFilteredBatch === 'All'
    ? students
    : students.filter(s => s.batch === currentFilteredBatch);

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="7"><div class="empty-state"><div class="empty-icon">👥</div>No students registered in batch "${currentFilteredBatch}"</div></td></tr>`;
    return;
  }
  tbody.innerHTML = filtered.map(s => `
    <tr>
      <td><strong>${s.name}</strong></td>
      <td><span class="mono">${s.student_id}</span></td>
      <td><span class="pill" style="background: rgba(56,139,253,.08); color: var(--blue); border: 1px solid var(--border); font-size:11px">${s.batch || '—'}</span></td>
      <td>${s.department || '—'}</td>
      <td>${s.email || '—'}</td>
      <td><span class="mono" style="font-size:11px">${fmtDate(s.registered_at)}</span></td>
      <td>
        <button class="btn btn-danger" style="padding:5px 12px;font-size:11px"
          onclick="deleteStudent('${s.student_id}','${s.name}')">✕ Remove</button>
      </td>
    </tr>`).join('');
}

function renderBatchTabs(students) {
  const container = document.getElementById('batchTabsContainer');
  if (!container) return;

  // Gather all unique batches from database students
  const uniqueBatches = new Set();
  students.forEach(s => {
    if (s.batch) uniqueBatches.add(s.batch);
  });

  // Standard preset batches
  const standardBatches = ['2020-22', '2021-23', '2022-24', '2023-25', '2024-26'];
  const databaseBatches = Array.from(uniqueBatches);

  // Merge unique batches from DB and standard list
  const allBatches = Array.from(new Set(['All', ...standardBatches, ...databaseBatches]));

  // Build the tabs markup
  container.innerHTML = allBatches.map(b => {
    const activeClass = b === currentFilteredBatch ? 'active' : '';
    // Count active students in this batch
    const count = b === 'All'
      ? students.length
      : students.filter(s => s.batch === b).length;

    return `<button type="button" class="batch-tab ${activeClass}" onclick="selectBatchFilter('${b}')">${b} (${count})</button>`;
  }).join('');
}

function selectBatchFilter(batch) {
  currentFilteredBatch = batch;
  renderStudentsTable(allRegisteredStudents);
}


async function deleteStudent(sid, name) {
  if (!confirm(`Remove ${name} from the system?`)) return;
  try {
    const r = await fetch(API + `/api/students/${sid}`, { method: 'DELETE' });
    const d = await r.json();
    toast(d.success ? '✅' : '❌', d.message, d.success ? 'success' : 'error');
    if (d.success) loadStudents();
  } catch(e) {}
}

// ── Registration ─────────────────────────────────────────────────────────────
let selectedPhoto = null;

function handlePhotoSelect(event) {
  const file = event.target.files[0];
  if (!file) return;
  selectedPhoto = file;
  const preview = document.getElementById('photoPreview');
  const icon = document.getElementById('dropIcon');
  const text = document.getElementById('dropText');
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.style.display = 'block';
    if (icon) icon.style.display = 'none';
    if (text) text.textContent = file.name;
  };
  reader.readAsDataURL(file);
}

async function registerStudent() {
  const sid   = document.getElementById('regId').value.trim();
  const name  = document.getElementById('regName').value.trim();
  const email = document.getElementById('regEmail').value.trim();
  const dept  = document.getElementById('regDept').value;
  const batch = document.getElementById('regBatch').value;

  if (!sid || !name) { toast('⚠️', 'Student ID and Name are required', 'error'); return; }
  if (!selectedPhoto) { toast('⚠️', 'Please select or capture a face photo', 'error'); return; }

  const btn = document.getElementById('regBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Registering...';

  try {
    const fd = new FormData();
    fd.append('student_id', sid);
    fd.append('name', name);
    fd.append('email', email);
    fd.append('department', dept);
    fd.append('batch', batch);
    fd.append('photo', selectedPhoto);

    const r = await fetch(API + '/api/students/register', { method: 'POST', body: fd });
    const d = await r.json();

    toast(d.success ? '✅' : '❌', d.message, d.success ? 'success' : 'error');

    if (d.success) {
      // Reset text fields
      ['regId','regName','regEmail'].forEach(id => { document.getElementById(id).value = ''; });
      document.getElementById('regDept').value = '';
      document.getElementById('regBatch').value = '';

      // Reset file upload previews
      document.getElementById('photoPreview').style.display = 'none';
      document.getElementById('photoPreview').src = '';
      document.getElementById('dropIcon').style.display = '';
      document.getElementById('dropText').textContent = 'Drop photo here or click to browse';

      // Reset webcam previews
      document.getElementById('regPhotoPreview').style.display = 'none';
      document.getElementById('regPhotoPreview').src = '';
      document.getElementById('regVideo').style.display = 'none';
      document.getElementById('webcamPlaceholder').style.display = 'flex';
      document.getElementById('regCaptureBtn').style.display = 'none';
      document.getElementById('regRetakeBtn').style.display = 'none';

      selectedPhoto = null;

      // Restart camera stream if they are still on webcam registration mode
      if (regPhotoMode === 'webcam') {
        startRegCamera();
      }

      refreshAll();
    }

  } catch(e) {
    toast('❌', 'Registration failed. Check server connection.', 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '✓ Register Student';
  }
}

// ── Export ───────────────────────────────────────────────────────────────────
function exportAttendance() {
  const date = document.getElementById('dateFilter')?.value;
  const url = date ? `${API}/api/attendance/export?date=${date}` : `${API}/api/attendance/export`;
  window.open(url, '_blank');
}

// ── Live Log ─────────────────────────────────────────────────────────────────
function addLiveLogItem(record) {
  const log1 = document.getElementById('liveLog');
  const log2 = document.getElementById('cameraLog');

  const initials = record.student_name
    .split(' ').map(n => n[0]).join('').toUpperCase().slice(0,2);

  const html = `
    <div class="log-item">
      <div class="log-avatar">${initials}</div>
      <div>
        <div class="log-name">${record.student_name}</div>
        <div class="log-time">${new Date().toLocaleTimeString()} · ${record.student_id}</div>
      </div>
      <span class="log-badge in">
        PRESENT
      </span>
    </div>`;

  [log1, log2].forEach(log => {
    if (!log) return;
    const empty = log.querySelector('.empty-state');
    if (empty) log.innerHTML = '';
    log.insertAdjacentHTML('afterbegin', html);
    // Keep max 20 items
    while (log.children.length > 20) log.removeChild(log.lastChild);
  });
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(icon, msg, type='') {
  const container = document.getElementById('toastContainer');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span style="font-size:16px">${icon}</span><span>${msg}</span>`;
  container.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function fmtTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
}

function fmtDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric'
  });
}

// Drag-and-drop on drop zone
const dz = document.getElementById('dropZone');
if (dz) {
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) { document.getElementById('photoInput').files = e.dataTransfer.files; handlePhotoSelect({target:{files:[f]}}); }
  });
}

// ── Authentication Controllers ──────────────────────────────────────────────
async function checkAuthStatus() {
  try {
    const res = await fetch('/api/auth/status');
    const data = await res.json();
    if (data.authenticated) {
      showMainSystem();
    } else {
      showLoginScreen();
    }
  } catch (err) {
    console.error("Auth status check failed:", err);
    showLoginScreen();
  }
}

function showMainSystem() {
  document.getElementById('loginWrapper').style.display = 'none';
  document.getElementById('mainLayout').style.display = 'grid';
  
  if (!socket) {
    connectSocket();
  }
  refreshAll();
}

function showLoginScreen() {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
  document.getElementById('loginWrapper').style.display = 'flex';
  document.getElementById('mainLayout').style.display = 'none';
  
  document.getElementById('loginUsername').value = '';
  document.getElementById('loginPassword').value = '';
  document.getElementById('loginError').style.display = 'none';
}

async function handleLogin(event) {
  event.preventDefault();
  const username = document.getElementById('loginUsername').value.trim();
  const password = document.getElementById('loginPassword').value.trim();
  const errorDiv = document.getElementById('loginError');
  const loginBtn = document.getElementById('loginBtn');
  
  errorDiv.style.display = 'none';
  loginBtn.disabled = true;
  loginBtn.textContent = 'Signing in...';
  
  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    
    const data = await res.json();
    if (res.ok && data.success) {
      showMainSystem();
      toast('🔑', 'Welcome back, Administrator!', 'success');
    } else {
      errorDiv.textContent = data.message || 'Invalid Admin ID or Password';
      errorDiv.style.display = 'block';
    }
  } catch (err) {
    errorDiv.textContent = 'Connection error. Please try again.';
    errorDiv.style.display = 'block';
  } finally {
    loginBtn.disabled = false;
    loginBtn.textContent = 'Sign In';
  }
}

async function handleLogout() {
  try {
    await fetch('/api/auth/logout', { method: 'POST' });
    showLoginScreen();
    toast('🚪', 'Logged out successfully');
  } catch (err) {
    console.error("Logout failed:", err);
    showLoginScreen();
  }
}

// ── Toggle Auth Views ────────────────────────────────────────────────────────
function toggleAuthView(view) {
  const loginView = document.getElementById('loginView');
  const registerView = document.getElementById('registerView');
  const subtitle = document.getElementById('loginCardSubtitle');
  
  if (view === 'register') {
    loginView.style.display = 'none';
    registerView.style.display = 'block';
    subtitle.textContent = 'Register a new Administrator account';
    
    document.getElementById('regUsername').value = '';
    document.getElementById('regPassword').value = '';
    document.getElementById('regConfirmPassword').value = '';
    document.getElementById('registerError').style.display = 'none';
  } else {
    loginView.style.display = 'block';
    registerView.style.display = 'none';
    subtitle.textContent = 'Smart FaceID Attendance System Login';
    
    document.getElementById('loginUsername').value = '';
    document.getElementById('loginPassword').value = '';
    document.getElementById('loginError').style.display = 'none';
  }
}

// ── Handle Account Registration ──────────────────────────────────────────────
async function handleRegister(event) {
  event.preventDefault();
  const username = document.getElementById('regUsername').value.trim();
  const password = document.getElementById('regPassword').value.trim();
  const confirmPassword = document.getElementById('regConfirmPassword').value.trim();
  const errorDiv = document.getElementById('registerError');
  const registerBtn = document.getElementById('registerBtn');
  
  errorDiv.style.display = 'none';
  
  if (password !== confirmPassword) {
    errorDiv.textContent = 'Passwords do not match';
    errorDiv.style.display = 'block';
    return;
  }
  
  if (password.length < 8) {
    errorDiv.textContent = 'Password must be at least 8 characters long';
    errorDiv.style.display = 'block';
    return;
  }

  if (!/[A-Z]/.test(password)) {
    errorDiv.textContent = 'Password must contain at least one uppercase letter';
    errorDiv.style.display = 'block';
    return;
  }

  if (!/[0-9]/.test(password)) {
    errorDiv.textContent = 'Password must contain at least one number';
    errorDiv.style.display = 'block';
    return;
  }

  if (!/[^a-zA-Z0-9]/.test(password)) {
    errorDiv.textContent = 'Password must contain at least one special character';
    errorDiv.style.display = 'block';
    return;
  }
  
  registerBtn.disabled = true;
  registerBtn.textContent = 'Creating account...';
  
  try {
    const res = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    
    const data = await res.json();
    if (res.ok && data.success) {
      toast('👤', 'Account created successfully!', 'success');
      toggleAuthView('login');
      document.getElementById('loginUsername').value = username;
      toast('🔑', 'Please sign in with your new credentials.');
    } else {
      errorDiv.textContent = data.message || 'Failed to create account';
      errorDiv.style.display = 'block';
    }
  } catch (err) {
    errorDiv.textContent = 'Connection error. Please try again.';
    errorDiv.style.display = 'block';
  } finally {
    registerBtn.disabled = false;
    registerBtn.textContent = 'Create Account';
  }
}

// ── Settings API Handlers ───────────────────────────────────────────────────
async function loadSettings() {
  try {
    const r = await fetch(API + '/api/settings');
    const d = await r.json();
    if (d.success) {
      document.getElementById('setLaplacian').value = d.laplacian_threshold;
      document.getElementById('setGlare').value = d.glare_threshold;
      document.getElementById('setTolerance').value = d.tolerance;
      if (d.scale_factor !== undefined) {
        document.getElementById('setScaleFactor').value = parseFloat(d.scale_factor).toFixed(2);
      }
    } else {
      toast('⚠️', 'Failed to load settings', 'error');
    }
  } catch(e) {
    console.error('Error loading settings:', e);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const laplacian_threshold = parseFloat(document.getElementById('setLaplacian').value);
  const glare_threshold = parseFloat(document.getElementById('setGlare').value);
  const tolerance = parseFloat(document.getElementById('setTolerance').value);
  const scale_factor = parseFloat(document.getElementById('setScaleFactor').value);

  try {
    const r = await fetch(API + '/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        laplacian_threshold,
        glare_threshold,
        tolerance,
        scale_factor
      })
    });
    const d = await r.json();
    if (d.success) {
      toast('⚙️', 'Settings updated successfully', 'success');
    } else {
      toast('⚠️', d.message || 'Failed to update settings', 'error');
    }
  } catch(e) {
    toast('❌', 'Error updating settings', 'error');
  }
}

// ── Theme Management ────────────────────────────────────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeUI(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  updateThemeUI(newTheme);
  toast('🌓', `Theme switched to ${newTheme} mode`, '');
}

function updateThemeUI(theme) {
  const loginToggle = document.getElementById('loginThemeToggle');
  const appToggle = document.getElementById('appThemeToggle');
  const icon = theme === 'light' ? '🌙' : '☀️';
  const title = theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode';
  
  if (loginToggle) {
    loginToggle.textContent = icon;
    loginToggle.title = title;
  }
  if (appToggle) {
    appToggle.textContent = icon;
    appToggle.title = title;
  }
}