from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pickle
import json

app = Flask(__name__)
CORS(app)

# Global ensemble model
ensemble_model = None
model_path = 'ensemble_model.pkl'

# KONFIGURASI THRESHOLD - Balance antara usability dan security
ENSEMBLE_THRESHOLD = 80.0      # Naikkan dari 75 → 80 (reduce false positive)
HAMMING_MIN_THRESHOLD = 70.0   # Naikkan dari 65 → 70 (lebih ketat)
COSINE_MIN_THRESHOLD = 65.0    # Naikkan dari 60 → 65 (lebih ketat)
MIN_TEMPLATE_LENGTH = 512
MIN_QUALITY_THRESHOLD = 0.20   # Sesuaikan dengan kemampuan sensor (20%)

# Multi-template configuration
REQUIRED_TEMPLATES = 3  # Jumlah template yang diperlukan per user

# Anti-duplicate check window (dalam detik)
DUPLICATE_CHECK_WINDOW = 10

# Inisialisasi database
def init_db():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    # Tabel users - UPDATED untuk multi-template
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    registered_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    templates_count INTEGER DEFAULT 0
                )''')
    
    # Tabel baru untuk menyimpan multiple templates per user
    c.execute('''CREATE TABLE IF NOT EXISTS user_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    template TEXT NOT NULL,
                    template_quality REAL,
                    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    template_index INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )''')
    
    # Tabel attendance logs
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    name TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    hamming_score REAL,
                    cosine_score REAL,
                    ensemble_score REAL,
                    matched_template_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )''')
    
    # Tabel untuk menyimpan verification attempts (untuk evaluasi)
    c.execute('''CREATE TABLE IF NOT EXISTS verification_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    template_scanned TEXT,
                    matched_user_id TEXT,
                    is_genuine INTEGER,
                    hamming_score REAL,
                    cosine_score REAL,
                    ensemble_score REAL,
                    passed_all_checks INTEGER
                )''')
    
    conn.commit()
    conn.close()

init_db()

# ===== SIMILARITY FUNCTIONS =====

def hex_to_bytes(hex_str):
    """Convert hex string to numpy array of bytes"""
    try:
        return np.array([int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)], dtype=np.uint8)
    except:
        return None

def validate_template(template_hex):
    """Validate template quality dengan threshold yang lebih realistis"""
    if not template_hex or len(template_hex) < MIN_TEMPLATE_LENGTH:
        return False, f"Template terlalu pendek ({len(template_hex)} < {MIN_TEMPLATE_LENGTH})"
    
    try:
        template_bytes = hex_to_bytes(template_hex)
        if template_bytes is None:
            return False, "Template tidak valid"
        
        # Check for all zeros or all 0xFF (bad quality)
        if np.all(template_bytes == 0) or np.all(template_bytes == 0xFF):
            return False, "Template quality buruk (semua nilai sama)"
        
        # Check entropy - template yang bagus harus memiliki variasi
        unique_ratio = len(np.unique(template_bytes)) / len(template_bytes)
        
        # THRESHOLD: 20% (disesuaikan dengan kemampuan sensor)
        if unique_ratio < MIN_QUALITY_THRESHOLD:
            return False, f"Template quality buruk (variasi: {unique_ratio:.2%}, minimum: {MIN_QUALITY_THRESHOLD:.0%})"
        
        return True, f"OK (quality: {unique_ratio:.2%})"
    except Exception as e:
        return False, f"Error validasi: {str(e)}"

def hamming_similarity(template1_hex, template2_hex):
    """Calculate Hamming similarity with validation"""
    if len(template1_hex) != len(template2_hex):
        return 0.0
    
    t1 = hex_to_bytes(template1_hex)
    t2 = hex_to_bytes(template2_hex)
    
    if t1 is None or t2 is None:
        return 0.0
    
    # XOR to find differing bits
    xor = np.bitwise_xor(t1, t2)
    # Count differing bits
    diff_bits = np.unpackbits(xor).sum()
    total_bits = len(t1) * 8
    
    # Return similarity (not distance)
    similarity = (1.0 - diff_bits / total_bits) * 100.0
    return float(similarity)

def cosine_similarity(template1_hex, template2_hex):
    """Calculate Cosine similarity with validation"""
    if len(template1_hex) != len(template2_hex):
        return 0.0
    
    t1 = hex_to_bytes(template1_hex)
    t2 = hex_to_bytes(template2_hex)
    
    if t1 is None or t2 is None:
        return 0.0
    
    t1 = t1.astype(np.float64)
    t2 = t2.astype(np.float64)
    
    # Cosine similarity
    dot_product = np.dot(t1, t2)
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_sim = dot_product / (norm1 * norm2)
    # Convert to percentage
    return float(cosine_sim * 100.0)

def compute_feature_vector(template1_hex, template2_hex):
    """Compute feature vector for ensemble model"""
    hamming = hamming_similarity(template1_hex, template2_hex)
    cosine = cosine_similarity(template1_hex, template2_hex)
    return np.array([hamming, cosine])

# ===== MULTI-STAGE VERIFICATION =====

def multi_stage_verification(hamming_score, cosine_score, ensemble_score):
    """
    Multi-stage verification untuk mengurangi false positive
    Returns: (passed, reason)
    """
    # Stage 1: Individual threshold check
    if hamming_score < HAMMING_MIN_THRESHOLD:
        return False, f"Hamming score terlalu rendah ({hamming_score:.2f}% < {HAMMING_MIN_THRESHOLD}%)"
    
    if cosine_score < COSINE_MIN_THRESHOLD:
        return False, f"Cosine score terlalu rendah ({cosine_score:.2f}% < {COSINE_MIN_THRESHOLD}%)"
    
    # Stage 2: Ensemble threshold
    if ensemble_score < ENSEMBLE_THRESHOLD:
        return False, f"Ensemble score terlalu rendah ({ensemble_score:.2f}% < {ENSEMBLE_THRESHOLD}%)"
    
    # Stage 3: Consistency check (DIPERKETAT untuk reduce false positive)
    score_diff = abs(hamming_score - cosine_score)
    if score_diff > 15.0:  # Perketat dari 25% → 15% untuk deteksi anomali
        return False, f"Skor tidak konsisten (selisih {score_diff:.2f}%)"
    
    # Stage 4: TAMBAHAN - Minimal average score
    avg_score = (hamming_score + cosine_score) / 2
    if avg_score < 68.0:  # Rata-rata harus > 68%
        return False, f"Average score terlalu rendah ({avg_score:.2f}% < 68%)"
    
    return True, "Passed all checks"

def check_duplicate_attendance(conn, user_id, window_seconds=DUPLICATE_CHECK_WINDOW):
    """Check if user already recorded attendance recently"""
    c = conn.cursor()
    time_threshold = datetime.now() - timedelta(seconds=window_seconds)
    
    c.execute("""SELECT timestamp FROM attendance 
                 WHERE user_id=? AND timestamp > ? 
                 ORDER BY timestamp DESC LIMIT 1""",
              (user_id, time_threshold))
    
    recent = c.fetchone()
    if recent:
        return True, recent[0]
    return False, None

# ===== ENSEMBLE MODEL TRAINING =====

def train_ensemble_model():
    """Train Logistic Regression ensemble model with class weights"""
    global ensemble_model
    
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    # Get all verification logs
    c.execute("""SELECT hamming_score, cosine_score, is_genuine 
                 FROM verification_log 
                 WHERE hamming_score IS NOT NULL AND cosine_score IS NOT NULL""")
    logs = c.fetchall()
    conn.close()
    
    if len(logs) < 30:
        print(f"⚠️ Not enough data for training ({len(logs)} samples). Need at least 30.")
        return False
    
    X = np.array([[log[0], log[1]] for log in logs])
    y = np.array([log[2] for log in logs])
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        print("⚠️ Need both genuine and impostor samples for training")
        return False
    
    # Train model with balanced class weights to reduce false positives
    ensemble_model = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        class_weight='balanced',
        C=0.5
    )
    ensemble_model.fit(X, y)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    print(f"✅ Ensemble model trained with {len(logs)} samples")
    print(f"   Genuine samples: {sum(y)}, Impostor samples: {len(y) - sum(y)}")
    
    return True

def load_ensemble_model():
    """Load pre-trained ensemble model"""
    global ensemble_model
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            ensemble_model = pickle.load(f)
        print("✅ Ensemble model loaded")
        return True
    return False

def predict_ensemble(hamming_score, cosine_score):
    """Predict match probability using ensemble"""
    if ensemble_model is None:
        # Fallback: weighted average
        return (hamming_score * 0.55 + cosine_score * 0.45)
    
    features = np.array([[hamming_score, cosine_score]])
    prob = ensemble_model.predict_proba(features)[0][1]
    return float(prob * 100.0)

# Load model on startup
load_ensemble_model()

# ===== WEB ROUTES =====

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/test', methods=['POST', 'GET'])
def test_endpoint():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            return jsonify({
                "status": "success",
                "message": "POST test OK",
                "received": data
            })
        except:
            return jsonify({
                "status": "error",
                "message": "JSON parse failed"
            }), 400
    else:
        return jsonify({
            "status": "success",
            "message": "GET test OK",
            "server": "Flask with Multi-Template Enrollment",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "required_templates": REQUIRED_TEMPLATES,
                "ensemble_threshold": ENSEMBLE_THRESHOLD,
                "hamming_threshold": HAMMING_MIN_THRESHOLD,
                "cosine_threshold": COSINE_MIN_THRESHOLD,
                "min_quality": MIN_QUALITY_THRESHOLD
            }
        })

# ===== API ENDPOINTS =====

@app.route('/api/register_user', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        name = data.get('name')
        
        if not user_id or not name:
            return jsonify({"status": "error", "message": "user_id dan name wajib diisi"}), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if c.fetchone():
            conn.close()
            return jsonify({"status": "error", "message": "User ID sudah terdaftar"}), 400
        
        c.execute("""INSERT INTO users (user_id, name, status, registered_at, templates_count) 
                     VALUES (?, ?, 'pending', ?, 0)""", 
                  (user_id, name, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"User {name} berhasil didaftarkan. Silakan scan {REQUIRED_TEMPLATES} kali untuk enrollment.",
            "user_id": user_id,
            "required_templates": REQUIRED_TEMPLATES
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/check_status/<user_id>', methods=['GET'])
def check_status(user_id):
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("SELECT status, templates_count FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({"status": "not_found"}), 404
        
        return jsonify({
            "status": user[0],
            "templates_count": user[1],
            "required_templates": REQUIRED_TEMPLATES
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/save_template', methods=['POST'])
def save_template():
    """Save template dengan support multi-template enrollment"""
    try:
        if not request.data:
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data = request.get_json(force=True)
        user_id = data.get('user_id') or data.get('id')
        template = data.get('template')
        
        if not user_id or not template:
            return jsonify({"status": "error", "message": "user_id dan template wajib diisi"}), 400
        
        # Validasi template quality
        is_valid, msg = validate_template(template)
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": f"Template tidak valid: {msg}"
            }), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("SELECT name, status, templates_count FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            return jsonify({
                "status": "error", 
                "message": f"User ID '{user_id}' tidak ditemukan. Silakan register terlebih dahulu."
            }), 404
        
        user_name, current_status, templates_count = user
        
        # Check if already fully enrolled
        if current_status == 'enrolled' and templates_count >= REQUIRED_TEMPLATES:
            conn.close()
            return jsonify({
                "status": "warning",
                "message": f"User {user_name} sudah fully enrolled dengan {templates_count} template"
            }), 200
        
        # Calculate template quality
        template_bytes = hex_to_bytes(template)
        quality = len(np.unique(template_bytes)) / len(template_bytes)
        
        # Save template
        new_index = templates_count + 1
        c.execute("""INSERT INTO user_templates 
                     (user_id, template, template_quality, template_index) 
                     VALUES (?, ?, ?, ?)""", 
                  (user_id, template, quality, new_index))
        
        # Update user templates count
        c.execute("""UPDATE users 
                     SET templates_count=?, status=? 
                     WHERE user_id=?""", 
                  (new_index, 
                   'enrolled' if new_index >= REQUIRED_TEMPLATES else 'enrolling',
                   user_id))
        
        conn.commit()
        conn.close()
        
        remaining = REQUIRED_TEMPLATES - new_index
        
        if remaining > 0:
            return jsonify({
                "status": "partial",
                "message": f"✅ Template {new_index}/{REQUIRED_TEMPLATES} untuk {user_name} tersimpan! Scan {remaining} kali lagi.",
                "user_id": user_id,
                "name": user_name,
                "quality": round(quality * 100, 2),
                "templates_count": new_index,
                "required_templates": REQUIRED_TEMPLATES,
                "remaining": remaining
            }), 200
        else:
            return jsonify({
                "status": "success",
                "message": f"🎉 Enrollment {user_name} SELESAI! Semua {REQUIRED_TEMPLATES} template tersimpan.",
                "user_id": user_id,
                "name": user_name,
                "quality": round(quality * 100, 2),
                "templates_count": new_index,
                "required_templates": REQUIRED_TEMPLATES
            }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/verify_template', methods=['POST'])
def verify_template():
    """Verify fingerprint menggunakan SEMUA template yang tersimpan"""
    try:
        data = request.get_json()
        scanned_template = data.get('template')
        
        if not scanned_template:
            return jsonify({"status": "error", "message": "template wajib diisi"}), 400
        
        # Validate scanned template
        is_valid, msg = validate_template(scanned_template)
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": f"Template scan tidak valid: {msg}"
            }), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        # Get all enrolled users with their templates
        c.execute("""SELECT u.user_id, u.name, ut.id, ut.template 
                     FROM users u
                     JOIN user_templates ut ON u.user_id = ut.user_id
                     WHERE u.status='enrolled'""")
        templates = c.fetchall()
        
        if not templates:
            conn.close()
            return jsonify({
                "status": "no_match", 
                "message": "Belum ada user yang fully enrolled"
            })
        
        # Compare with ALL stored templates
        best_match = None
        best_ensemble_score = 0.0
        best_hamming = 0.0
        best_cosine = 0.0
        best_passed_checks = False
        best_template_id = None
        
        user_best_scores = {}  # Track best score per user
        
        for template_data in templates:
            user_id, name, template_id, stored_template = template_data
            
            # Calculate similarities
            hamming = hamming_similarity(stored_template, scanned_template)
            cosine = cosine_similarity(stored_template, scanned_template)
            ensemble = predict_ensemble(hamming, cosine)
            
            # Multi-stage verification
            passed_checks, check_reason = multi_stage_verification(hamming, cosine, ensemble)
            
            # Track best score for this user (across all their templates)
            if user_id not in user_best_scores or ensemble > user_best_scores[user_id]['ensemble']:
                user_best_scores[user_id] = {
                    'name': name,
                    'hamming': hamming,
                    'cosine': cosine,
                    'ensemble': ensemble,
                    'passed_checks': passed_checks,
                    'check_reason': check_reason,
                    'template_id': template_id
                }
            
            # Update global best match
            if passed_checks and ensemble > best_ensemble_score:
                best_ensemble_score = ensemble
                best_hamming = hamming
                best_cosine = cosine
                best_match = (user_id, name)
                best_passed_checks = True
                best_template_id = template_id
        
        # Prepare results per user
        results = []
        for uid, scores in user_best_scores.items():
            results.append({
                'user_id': uid,
                'name': scores['name'],
                'hamming': round(scores['hamming'], 2),
                'cosine': round(scores['cosine'], 2),
                'ensemble': round(scores['ensemble'], 2),
                'passed_checks': scores['passed_checks'],
                'check_reason': scores['check_reason']
            })
        
        if best_match and best_passed_checks:
            user_id, name = best_match
            
            # Check for duplicate attendance
            is_duplicate, last_time = check_duplicate_attendance(conn, user_id)
            if is_duplicate:
                conn.close()
                return jsonify({
                    "status": "duplicate",
                    "message": f"⚠️ {name} sudah absen baru-baru ini ({last_time})",
                    "user_id": user_id,
                    "name": name,
                    "last_attendance": last_time
                })
            
            # Log verification (genuine match)
            c.execute("""INSERT INTO verification_log 
                        (template_scanned, matched_user_id, is_genuine, hamming_score, cosine_score, ensemble_score, passed_all_checks)
                        VALUES (?, ?, 1, ?, ?, ?, 1)""",
                     (scanned_template[:100], user_id, best_hamming, best_cosine, best_ensemble_score))
            
            # Record attendance
            c.execute("""INSERT INTO attendance 
                        (user_id, name, confidence, hamming_score, cosine_score, ensemble_score, matched_template_id) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)""", 
                     (user_id, name, best_ensemble_score, best_hamming, best_cosine, best_ensemble_score, best_template_id))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                "status": "match",
                "user_id": user_id,
                "name": name,
                "scores": {
                    "hamming": round(best_hamming, 2),
                    "cosine": round(best_cosine, 2),
                    "ensemble": round(best_ensemble_score, 2)
                },
                "matched_template_id": best_template_id,
                "all_results": results,
                "message": f"✅ Selamat datang, {name}!"
            })
        else:
            # Log verification (impostor or failed checks)
            c.execute("""INSERT INTO verification_log 
                        (template_scanned, matched_user_id, is_genuine, hamming_score, cosine_score, ensemble_score, passed_all_checks)
                        VALUES (?, NULL, 0, ?, ?, ?, 0)""",
                     (scanned_template[:100], best_hamming, best_cosine, best_ensemble_score))
            conn.commit()
            conn.close()
            
            return jsonify({
                "status": "no_match",
                "message": "❌ Sidik jari tidak terdaftar atau tidak memenuhi kriteria",
                "best_scores": {
                    "hamming": round(best_hamming, 2),
                    "cosine": round(best_cosine, 2),
                    "ensemble": round(best_ensemble_score, 2)
                },
                "all_results": results,
                "thresholds": {
                    "ensemble": ENSEMBLE_THRESHOLD,
                    "hamming": HAMMING_MIN_THRESHOLD,
                    "cosine": COSINE_MIN_THRESHOLD
                }
            })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/record_attendance', methods=['POST'])
def record_attendance():
    """Simplified attendance recording"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        name = data.get('name')
        confidence = data.get('confidence', 0)
        
        conn = sqlite3.connect('fingerprint.db')
        
        # Check duplicate
        is_duplicate, last_time = check_duplicate_attendance(conn, user_id)
        if is_duplicate:
            conn.close()
            return jsonify({
                "status": "duplicate",
                "message": f"Already recorded at {last_time}"
            })
        
        c = conn.cursor()
        c.execute("""INSERT INTO attendance (user_id, name, confidence) 
                     VALUES (?, ?, ?)""", (user_id, name, confidence))
        
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "Attendance recorded"})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/delete_user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete user dan semua templatenya"""
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        # Delete templates first (foreign key cascade should handle this, but explicit is better)
        c.execute("DELETE FROM user_templates WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "User and all templates deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users dengan info template mereka"""
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    c.execute("""SELECT u.user_id, u.name, u.status, u.registered_at, u.templates_count,
                        GROUP_CONCAT(ut.template_quality) as qualities
                 FROM users u
                 LEFT JOIN user_templates ut ON u.user_id = ut.user_id
                 GROUP BY u.user_id""")
    users = c.fetchall()
    conn.close()
    
    user_list = []
    for user in users:
        qualities = []
        if user[5]:  # if qualities exist
            qualities = [round(float(q) * 100, 2) for q in user[5].split(',')]
        
        user_list.append({
            "user_id": user[0],
            "name": user[1],
            "status": user[2],
            "registered_at": user[3],
            "templates_count": user[4],
            "required_templates": REQUIRED_TEMPLATES,
            "template_qualities": qualities,
            "avg_quality": round(sum(qualities) / len(qualities), 2) if qualities else None
        })
    
    return jsonify({"status": "success", "users": user_list})

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    c.execute("""SELECT user_id, name, timestamp, confidence, hamming_score, cosine_score, ensemble_score, matched_template_id
                 FROM attendance 
                 ORDER BY timestamp DESC 
                 LIMIT 50""")
    logs = c.fetchall()
    conn.close()
    
    attendance_list = []
    for log in logs:
        attendance_list.append({
            "user_id": log[0],
            "name": log[1],
            "timestamp": log[2],
            "confidence": log[3],
            "hamming_score": log[4],
            "cosine_score": log[5],
            "ensemble_score": log[6],
            "matched_template_id": log[7]
        })
    
    return jsonify({"status": "success", "attendance": attendance_list})

# ===== EVALUATION ENDPOINTS =====

@app.route('/api/train_ensemble', methods=['POST'])
def train_ensemble_endpoint():
    """Train ensemble model"""
    success = train_ensemble_model()
    if success:
        return jsonify({"status": "success", "message": "Model trained successfully"})
    else:
        return jsonify({"status": "error", "message": "Not enough training data"}), 400

@app.route('/api/evaluate', methods=['GET'])
def evaluate_model():
    """Evaluate model and compute ROC, EER, AUC"""
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("""SELECT hamming_score, cosine_score, ensemble_score, is_genuine 
                     FROM verification_log 
                     WHERE hamming_score IS NOT NULL""")
        logs = c.fetchall()
        conn.close()
        
        if len(logs) < 10:
            return jsonify({
                "status": "error",
                "message": f"Not enough data for evaluation ({len(logs)} samples)"
            }), 400
        
        # Prepare data
        hamming_scores = np.array([log[0] for log in logs])
        cosine_scores = np.array([log[1] for log in logs])
        ensemble_scores = np.array([log[2] for log in logs])
        y_true = np.array([log[3] for log in logs])
        
        results = {}
        
        # Evaluate each method
        for name, scores in [('Hamming', hamming_scores), 
                             ('Cosine', cosine_scores), 
                             ('Ensemble', ensemble_scores)]:
            
            # Compute ROC
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            
            # Compute EER
            fnr = 1 - tpr
            eer_index = np.nanargmin(np.abs(fpr - fnr))
            eer = (fpr[eer_index] + fnr[eer_index]) / 2
            eer_threshold = thresholds[eer_index]
            
            # Find threshold at FAR = 0.1% and 1%
            far_001_idx = np.where(fpr <= 0.001)[0]
            far_01_idx = np.where(fpr <= 0.01)[0]
            
            threshold_far_001 = thresholds[far_001_idx[-1]] if len(far_001_idx) > 0 else None
            threshold_far_01 = thresholds[far_01_idx[-1]] if len(far_01_idx) > 0 else None
            
            results[name] = {
                'AUC': float(roc_auc),
                'EER': float(eer * 100),
                'EER_threshold': float(eer_threshold),
                'threshold_FAR_0.1%': float(threshold_far_001) if threshold_far_001 else None,
                'threshold_FAR_1%': float(threshold_far_01) if threshold_far_01 else None,
                'ROC': {
                    'FPR': fpr.tolist(),
                    'TPR': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            }
        
        # Statistics
        genuine_count = sum(y_true)
        impostor_count = len(y_true) - genuine_count
        
        return jsonify({
            "status": "success",
            "total_samples": len(logs),
            "genuine_samples": int(genuine_count),
            "impostor_samples": int(impostor_count),
            "results": results
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        "status": "success",
        "config": {
            "REQUIRED_TEMPLATES": REQUIRED_TEMPLATES,
            "ENSEMBLE_THRESHOLD": ENSEMBLE_THRESHOLD,
            "HAMMING_MIN_THRESHOLD": HAMMING_MIN_THRESHOLD,
            "COSINE_MIN_THRESHOLD": COSINE_MIN_THRESHOLD,
            "MIN_TEMPLATE_LENGTH": MIN_TEMPLATE_LENGTH,
            "MIN_QUALITY_THRESHOLD": MIN_QUALITY_THRESHOLD,
            "DUPLICATE_CHECK_WINDOW": DUPLICATE_CHECK_WINDOW
        }
    })

@app.route('/api/user/<user_id>/templates', methods=['GET'])
def get_user_templates(user_id):
    """Get all templates for a specific user"""
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("""SELECT id, template_quality, enrolled_at, template_index 
                     FROM user_templates 
                     WHERE user_id=? 
                     ORDER BY template_index""", (user_id,))
        templates = c.fetchall()
        conn.close()
        
        if not templates:
            return jsonify({
                "status": "error",
                "message": "User not found or has no templates"
            }), 404
        
        template_list = []
        for t in templates:
            template_list.append({
                "id": t[0],
                "quality": round(t[1] * 100, 2),
                "enrolled_at": t[2],
                "index": t[3]
            })
        
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "templates": template_list,
            "total": len(template_list)
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)