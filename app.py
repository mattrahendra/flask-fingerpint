from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime
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

# ===== IMPROVED THRESHOLDS =====
VERIFICATION_THRESHOLD = 80.0  # Naikkan dari 70% ke 80%
QUALITY_THRESHOLD = 0.25  # Minimum quality score
MIN_HAMMING_SCORE = 75.0  # Minimum Hamming untuk genuine
MIN_COSINE_SCORE = 70.0   # Minimum Cosine untuk genuine

# Inisialisasi database
def init_db():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    # Tabel users dengan multiple templates support
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    template1 TEXT,
                    template2 TEXT,
                    template3 TEXT,
                    registered_at TIMESTAMP,
                    enrolled_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    template_count INTEGER DEFAULT 0
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
                    quality_score REAL,
                    matched_template INTEGER,
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
                    quality_score REAL,
                    decision TEXT
                )''')
    
    conn.commit()
    conn.close()

init_db()

# ===== IMPROVED SIMILARITY FUNCTIONS =====

def hex_to_bytes(hex_str):
    """Convert hex string to numpy array of bytes"""
    return np.array([int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)], dtype=np.uint8)

def check_template_quality(template_hex):
    """
    Check template quality to reject low-quality scans
    Returns: (is_valid, quality_score, message)
    """
    if len(template_hex) < 1000:
        return False, 0.0, "Template too short"
    
    t = hex_to_bytes(template_hex)
    
    # Check uniqueness (variety of byte values)
    unique_ratio = len(np.unique(t)) / len(t)
    
    # Check bit distribution (should be balanced)
    bits = np.unpackbits(t)
    ones_ratio = bits.sum() / len(bits)
    
    # Check for excessive repetition
    diffs = np.diff(t.astype(np.int16))
    zero_diffs = np.sum(diffs == 0) / len(diffs)
    
    # Calculate quality score (0-1)
    quality_score = 0.0
    
    # Good unique ratio: 0.3-0.9
    if 0.3 <= unique_ratio <= 0.9:
        quality_score += 0.4
    else:
        quality_score += 0.4 * (unique_ratio / 0.9 if unique_ratio < 0.9 else 0.9 / unique_ratio)
    
    # Good ones ratio: 0.3-0.7 (balanced bits)
    if 0.3 <= ones_ratio <= 0.7:
        quality_score += 0.3
    else:
        distance = min(abs(ones_ratio - 0.3), abs(ones_ratio - 0.7))
        quality_score += 0.3 * (1 - distance)
    
    # Low repetition is good
    if zero_diffs < 0.3:
        quality_score += 0.3
    else:
        quality_score += 0.3 * (1 - zero_diffs)
    
    is_valid = quality_score >= QUALITY_THRESHOLD
    
    message = f"Quality: {quality_score:.2f} (Unique: {unique_ratio:.2f}, Bits: {ones_ratio:.2f}, Rep: {zero_diffs:.2f})"
    
    return is_valid, quality_score, message

def weighted_zone_hamming(template1_hex, template2_hex):
    """
    Weighted Hamming similarity with zone-based matching
    Core region (center) is weighted more heavily
    """
    if len(template1_hex) != len(template2_hex):
        return 0.0
    
    t1 = hex_to_bytes(template1_hex)
    t2 = hex_to_bytes(template2_hex)
    
    # Divide into 3 zones: core (50%), mid (30%), edge (20%)
    total_len = len(t1)
    zone_splits = [
        (int(total_len * 0.25), int(total_len * 0.75), 0.5),  # Core zone (middle 50%)
        (0, int(total_len * 0.25), 0.25),                     # Start zone
        (int(total_len * 0.75), total_len, 0.25)              # End zone
    ]
    
    total_similarity = 0.0
    
    for start, end, weight in zone_splits:
        zone1 = t1[start:end]
        zone2 = t2[start:end]
        
        xor = np.bitwise_xor(zone1, zone2)
        diff_bits = np.unpackbits(xor).sum()
        total_bits = len(zone1) * 8
        
        zone_similarity = (1.0 - diff_bits / total_bits)
        total_similarity += zone_similarity * weight
    
    return float(total_similarity * 100.0)

def hamming_similarity(template1_hex, template2_hex):
    """Standard Hamming similarity (kept for backward compatibility)"""
    if len(template1_hex) != len(template2_hex):
        return 0.0
    
    t1 = hex_to_bytes(template1_hex)
    t2 = hex_to_bytes(template2_hex)
    
    xor = np.bitwise_xor(t1, t2)
    diff_bits = np.unpackbits(xor).sum()
    total_bits = len(t1) * 8
    
    similarity = (1.0 - diff_bits / total_bits) * 100.0
    return float(similarity)

def cosine_similarity(template1_hex, template2_hex):
    """Calculate Cosine similarity"""
    if len(template1_hex) != len(template2_hex):
        return 0.0
    
    t1 = hex_to_bytes(template1_hex).astype(np.float64)
    t2 = hex_to_bytes(template2_hex).astype(np.float64)
    
    dot_product = np.dot(t1, t2)
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_sim = dot_product / (norm1 * norm2)
    return float(cosine_sim * 100.0)

def compute_feature_vector(template1_hex, template2_hex):
    """Compute enhanced feature vector for ensemble model"""
    hamming = hamming_similarity(template1_hex, template2_hex)
    weighted_hamming = weighted_zone_hamming(template1_hex, template2_hex)
    cosine = cosine_similarity(template1_hex, template2_hex)
    
    # Additional features
    return np.array([hamming, weighted_hamming, cosine])

# ===== ENSEMBLE MODEL TRAINING =====

def train_ensemble_model():
    """Train Logistic Regression ensemble model with improved features"""
    global ensemble_model
    
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    c.execute("""SELECT hamming_score, cosine_score, is_genuine 
                 FROM verification_log 
                 WHERE hamming_score IS NOT NULL AND cosine_score IS NOT NULL""")
    logs = c.fetchall()
    conn.close()
    
    if len(logs) < 20:
        print(f"⚠️ Not enough data for training ({len(logs)} samples). Need at least 20.")
        return False
    
    X = np.array([[log[0], log[0] * 1.1, log[1]] for log in logs])  # Simulate weighted hamming
    y = np.array([log[2] for log in logs])
    
    if len(np.unique(y)) < 2:
        print("⚠️ Need both genuine and impostor samples for training")
        return False
    
    ensemble_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    ensemble_model.fit(X, y)
    
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
    """Predict match probability using ensemble with quality checks"""
    if ensemble_model is None:
        # Fallback: weighted average with strict thresholds
        weighted_score = (hamming_score * 0.6 + cosine_score * 0.4)
        return weighted_score
    
    weighted_hamming = hamming_score * 1.1  # Simulate weighted
    features = np.array([[hamming_score, weighted_hamming, cosine_score]])
    prob = ensemble_model.predict_proba(features)[0][1]
    return float(prob * 100.0)

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
            "total_samples": len(logs),
            "genuine_samples": int(genuine_count),
            "impostor_samples": int(impostor_count),
            "results": results,
            "current_thresholds": {
                "verification": VERIFICATION_THRESHOLD,
                "min_hamming": MIN_HAMMING_SCORE,
                "min_cosine": MIN_COSINE_SCORE,
                "quality": QUALITY_THRESHOLD
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        # User stats
        c.execute("SELECT COUNT(*) FROM users")
        total_users = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM users WHERE status='enrolled'")
        enrolled = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM users WHERE status='pending'")
        pending = c.fetchone()[0]
        
        # Attendance stats
        c.execute("SELECT COUNT(*) FROM attendance")
        total_attendance = c.fetchone()[0]
        
        # Verification stats
        c.execute("SELECT COUNT(*) FROM verification_log WHERE is_genuine=1")
        genuine_attempts = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM verification_log WHERE is_genuine=0")
        impostor_attempts = c.fetchone()[0]
        
        # Average scores
        c.execute("""SELECT AVG(ensemble_score), AVG(quality_score) 
                     FROM verification_log WHERE is_genuine=1""")
        avg_genuine = c.fetchone()
        
        c.execute("""SELECT AVG(ensemble_score), AVG(quality_score) 
                     FROM verification_log WHERE is_genuine=0""")
        avg_impostor = c.fetchone()
        
        conn.close()
        
        return jsonify({
            "status": "success",
            "users": {
                "total": total_users,
                "enrolled": enrolled,
                "pending": pending
            },
            "attendance": {
                "total": total_attendance
            },
            "verification": {
                "genuine_attempts": genuine_attempts,
                "impostor_attempts": impostor_attempts,
                "total_attempts": genuine_attempts + impostor_attempts,
                "avg_genuine_score": round(avg_genuine[0], 2) if avg_genuine[0] else None,
                "avg_genuine_quality": round(avg_genuine[1], 2) if avg_genuine[1] else None,
                "avg_impostor_score": round(avg_impostor[0], 2) if avg_impostor[0] else None,
                "avg_impostor_quality": round(avg_impostor[1], 2) if avg_impostor[1] else None
            },
            "thresholds": {
                "verification": VERIFICATION_THRESHOLD,
                "min_hamming": MIN_HAMMING_SCORE,
                "min_cosine": MIN_COSINE_SCORE,
                "quality": QUALITY_THRESHOLD
            }
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) "success",
            "message": "GET test OK",
            "server": "Flask with Improved Ensemble Matcher v2.0",
            "timestamp": datetime.now().isoformat(),
            "threshold": VERIFICATION_THRESHOLD
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
        
        c.execute("""INSERT INTO users (user_id, name, status, registered_at, template_count) 
                     VALUES (?, ?, 'pending', ?, 0)""", 
                  (user_id, name, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"User {name} berhasil didaftarkan. Silakan scan fingerprint di ESP32.",
            "user_id": user_id
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/check_status/<user_id>', methods=['GET'])
def check_status(user_id):
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("SELECT status, template_count FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({"status": "not_found"}), 404
        
        return jsonify({
            "status": user[0],
            "template_count": user[1]
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/save_template', methods=['POST'])
def save_template():
    """Save template with quality check"""
    try:
        if not request.data:
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data = request.get_json(force=True)
        user_id = data.get('user_id') or data.get('id')
        template = data.get('template')
        
        if not user_id or not template:
            return jsonify({"status": "error", "message": "user_id dan template wajib diisi"}), 400
        
        if len(template) < 100:
            return jsonify({
                "status": "error",
                "message": f"Template terlalu pendek ({len(template)} chars)"
            }), 400
        
        # Quality check
        is_valid, quality_score, quality_msg = check_template_quality(template)
        
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": f"Template quality terlalu rendah: {quality_msg}",
                "quality_score": round(quality_score, 2)
            }), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("SELECT name, status, template_count, template1, template2 FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            return jsonify({
                "status": "error", 
                "message": f"User ID '{user_id}' tidak ditemukan"
            }), 404
        
        user_name, current_status, template_count, template1, template2 = user
        
        # Save to appropriate template slot
        if template_count == 0:
            c.execute("""UPDATE users 
                         SET template1=?, status='enrolled', enrolled_at=?, template_count=1 
                         WHERE user_id=?""", 
                      (template, datetime.now(), user_id))
            slot = 1
        elif template_count == 1:
            c.execute("""UPDATE users 
                         SET template2=?, template_count=2 
                         WHERE user_id=?""", 
                      (template, user_id))
            slot = 2
        elif template_count == 2:
            c.execute("""UPDATE users 
                         SET template3=?, template_count=3 
                         WHERE user_id=?""", 
                      (template, user_id))
            slot = 3
        else:
            conn.close()
            return jsonify({
                "status": "error",
                "message": f"User sudah memiliki 3 templates"
            }), 400
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"Template {slot} untuk {user_name} berhasil disimpan!",
            "user_id": user_id,
            "name": user_name,
            "quality_score": round(quality_score, 2),
            "quality_message": quality_msg,
            "template_count": slot
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/verify_template', methods=['POST'])
def verify_template():
    """Improved verification with multiple templates and strict thresholds"""
    try:
        data = request.get_json()
        scanned_template = data.get('template')
        
        if not scanned_template:
            return jsonify({"status": "error", "message": "template wajib diisi"}), 400
        
        # Quality check on scanned template
        is_valid, quality_score, quality_msg = check_template_quality(scanned_template)
        
        if not is_valid:
            return jsonify({
                "status": "no_match",
                "message": f"⚠️ Template quality rendah: {quality_msg}",
                "quality_score": round(quality_score, 2)
            })
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        # Get all enrolled users with their templates
        c.execute("""SELECT user_id, name, template1, template2, template3, template_count 
                     FROM users WHERE status='enrolled'""")
        users = c.fetchall()
        
        if not users:
            conn.close()
            return jsonify({"status": "no_match", "message": "Belum ada user yang terdaftar"})
        
        # Compare with all enrolled users and all their templates
        best_match = None
        best_ensemble_score = 0.0
        best_hamming = 0.0
        best_cosine = 0.0
        best_template_slot = 0
        
        results = []
        
        for user in users:
            user_id, name, template1, template2, template3, template_count = user
            
            templates = [t for t in [template1, template2, template3] if t is not None]
            
            for idx, stored_template in enumerate(templates, 1):
                # Calculate similarities
                hamming = hamming_similarity(stored_template, scanned_template)
                weighted_hamming = weighted_zone_hamming(stored_template, scanned_template)
                cosine = cosine_similarity(stored_template, scanned_template)
                ensemble = predict_ensemble(hamming, cosine)
                
                # Strict filtering: both Hamming and Cosine must pass minimum
                if hamming >= MIN_HAMMING_SCORE and cosine >= MIN_COSINE_SCORE:
                    results.append({
                        'user_id': user_id,
                        'name': name,
                        'template_slot': idx,
                        'hamming': hamming,
                        'weighted_hamming': weighted_hamming,
                        'cosine': cosine,
                        'ensemble': ensemble
                    })
                    
                    if ensemble > best_ensemble_score:
                        best_ensemble_score = ensemble
                        best_hamming = hamming
                        best_cosine = cosine
                        best_template_slot = idx
                        best_match = (user_id, name)
        
        # Apply threshold
        if best_match and best_ensemble_score >= VERIFICATION_THRESHOLD:
            user_id, name = best_match
            
            # Determine confidence level
            if best_ensemble_score >= 95:
                confidence_level = "Very High"
            elif best_ensemble_score >= 90:
                confidence_level = "High"
            elif best_ensemble_score >= 85:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            # Log verification (genuine match)
            c.execute("""INSERT INTO verification_log 
                        (template_scanned, matched_user_id, is_genuine, hamming_score, cosine_score, ensemble_score, quality_score, decision)
                        VALUES (?, ?, 1, ?, ?, ?, ?, 'ACCEPT')""",
                     (scanned_template[:100], user_id, best_hamming, best_cosine, best_ensemble_score, quality_score))
            
            # Record attendance
            c.execute("""INSERT INTO attendance 
                        (user_id, name, confidence, hamming_score, cosine_score, ensemble_score, quality_score, matched_template) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                     (user_id, name, best_ensemble_score, best_hamming, best_cosine, best_ensemble_score, quality_score, best_template_slot))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                "status": "match",
                "user_id": user_id,
                "name": name,
                "confidence_level": confidence_level,
                "matched_template": best_template_slot,
                "scores": {
                    "hamming": round(best_hamming, 2),
                    "cosine": round(best_cosine, 2),
                    "ensemble": round(best_ensemble_score, 2),
                    "quality": round(quality_score, 2)
                },
                "all_results": results[:5],  # Top 5 results
                "message": f"✅ Selamat datang, {name}! (Confidence: {confidence_level})"
            })
        else:
            # Log verification (impostor)
            c.execute("""INSERT INTO verification_log 
                        (template_scanned, matched_user_id, is_genuine, hamming_score, cosine_score, ensemble_score, quality_score, decision)
                        VALUES (?, NULL, 0, ?, ?, ?, ?, 'REJECT')""",
                     (scanned_template[:100], best_hamming, best_cosine, best_ensemble_score, quality_score))
            conn.commit()
            conn.close()
            
            return jsonify({
                "status": "no_match",
                "message": "❌ Sidik jari tidak terdaftar atau confidence terlalu rendah",
                "best_scores": {
                    "hamming": round(best_hamming, 2),
                    "cosine": round(best_cosine, 2),
                    "ensemble": round(best_ensemble_score, 2),
                    "quality": round(quality_score, 2)
                },
                "all_results": results[:3],
                "threshold": VERIFICATION_THRESHOLD,
                "min_thresholds": {
                    "hamming": MIN_HAMMING_SCORE,
                    "cosine": MIN_COSINE_SCORE
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
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE user_id=? AND status='pending'", (user_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "User deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    c.execute("""SELECT user_id, name, status, registered_at, enrolled_at, 
                 template1, template_count FROM users""")
    users = c.fetchall()
    conn.close()
    
    user_list = []
    for user in users:
        user_list.append({
            "user_id": user[0],
            "name": user[1],
            "status": user[2],
            "registered_at": user[3],
            "enrolled_at": user[4],
            "template": user[5],  # Only send first template for ESP32
            "template_count": user[6]
        })
    
    return jsonify({"status": "success", "users": user_list})

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    c.execute("""SELECT user_id, name, timestamp, confidence, hamming_score, cosine_score, 
                 ensemble_score, quality_score, matched_template
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
            "quality_score": log[7],
            "matched_template": log[8]
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
        
        hamming_scores = np.array([log[0] for log in logs])
        cosine_scores = np.array([log[1] for log in logs])
        ensemble_scores = np.array([log[2] for log in logs])
        y_true = np.array([log[3] for log in logs])
        
        results = {}
        
        for name, scores in [('Hamming', hamming_scores), 
                             ('Cosine', cosine_scores), 
                             ('Ensemble', ensemble_scores)]:
            
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            
            fnr = 1 - tpr
            eer_index = np.nanargmin(np.abs(fpr - fnr))
            eer = (fpr[eer_index] + fnr[eer_index]) / 2
            eer_threshold = thresholds[eer_index]
            
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
        
        genuine_count = sum(y_true)
        impostor_count = len(y_true) - genuine_count
        
        return jsonify({
            "status": "success",
            "message": "Model evaluation completed",
            "results": results,
            "genuine_count": genuine_count,
            "impostor_count": impostor_count
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 
