from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime, timedelta
import pytz
import threading
import time

app = Flask(__name__)
CORS(app)

# Timezone Indonesia (WIB)
WIB = pytz.timezone('Asia/Jakarta')

# Anti-duplicate check window (dalam detik)
DUPLICATE_CHECK_WINDOW = 10

# Database connection dengan timeout dan configuration
def get_db_connection():
    conn = sqlite3.connect('fingerprint.db', timeout=30.0)
    conn.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging untuk concurrent access
    conn.execute('PRAGMA busy_timeout=30000')  # 30 detik timeout
    conn.row_factory = sqlite3.Row
    return conn

# Thread lock untuk operasi write
db_lock = threading.Lock()

# Inisialisasi database
def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Tabel users (single template per user)
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    registered_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    template TEXT
                )''')
    
    # Tabel access logs
    c.execute('''CREATE TABLE IF NOT EXISTS access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    name TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success INTEGER,
                    confidence INTEGER,
                    sensor_id INTEGER,
                    access_type TEXT DEFAULT 'verify'
                )''')
    
    # Tabel mapping sensor_slot_id ke user_id
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_mapping (
                    sensor_slot_id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )''')
    
    # Tabel enrollment status (untuk real-time feedback)
    c.execute('''CREATE TABLE IF NOT EXISTS enrollment_status (
                    user_id TEXT PRIMARY KEY,
                    status TEXT,
                    message TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )''')
    
    conn.commit()
    conn.close()

init_db()

def get_wib_time():
    """Get current time in WIB timezone"""
    return datetime.now(WIB)

def check_duplicate_access(conn, user_id, window_seconds=DUPLICATE_CHECK_WINDOW):
    """Check if user already accessed recently"""
    c = conn.cursor()
    time_threshold = get_wib_time() - timedelta(seconds=window_seconds)
    
    c.execute("""SELECT timestamp FROM access_logs 
                 WHERE user_id=? AND success=1 AND timestamp > ? 
                 ORDER BY timestamp DESC LIMIT 1""",
              (user_id, time_threshold))
    
    recent = c.fetchone()
    if recent:
        return True, recent[0]
    return False, None

# ===== WEB ROUTES =====

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        "status": "success",
        "message": "Server OK - Enhanced Dual-Scan Enrollment",
        "timestamp": get_wib_time().isoformat(),
        "timezone": "Asia/Jakarta (WIB)"
    })

# ===== ENROLLMENT API =====

@app.route('/api/start_enroll', methods=['POST'])
def start_enroll():
    """Inisialisasi enrollment baru"""
    try:
        data = request.get_json()
        name = data.get('name')
        
        if not name:
            return jsonify({"status": "error", "message": "Name is required"}), 400
        
        with db_lock:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Generate user_id otomatis
            c.execute("SELECT MAX(CAST(user_id AS INTEGER)) FROM users WHERE user_id GLOB '[0-9]*'")
            result = c.fetchone()
            next_id = 1 if not result[0] else int(result[0]) + 1
            user_id = str(next_id).zfill(3)
            
            c.execute("""INSERT INTO users (user_id, name, status, registered_at) 
                         VALUES (?, ?, 'pending', ?)""", 
                      (user_id, name, get_wib_time()))
            
            # Initialize enrollment status
            c.execute("""INSERT INTO enrollment_status (user_id, status, message, updated_at)
                         VALUES (?, 'pending', 'Waiting for ESP32 to detect', ?)""",
                      (user_id, get_wib_time()))
            
            conn.commit()
            conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"Enrollment started for {name}",
            "user_id": user_id,
            "name": name
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/enroll_status', methods=['POST'])
def update_enroll_status():
    """Update status enrollment dari ESP32"""
    try:
        data = request.get_json(force=True)
        user_id = data.get('user_id')
        status = data.get('status')
        message = data.get('message')
        
        if not user_id:
            return jsonify({"status": "error", "message": "user_id required"}), 400
        
        with db_lock:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Update enrollment status
            c.execute("""INSERT OR REPLACE INTO enrollment_status 
                         (user_id, status, message, updated_at)
                         VALUES (?, ?, ?, ?)""",
                      (user_id, status, message, get_wib_time()))
            
            conn.commit()
            conn.close()
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/check_enroll_status/<user_id>', methods=['GET'])
def check_enroll_status(user_id):
    """Check status enrollment untuk web interface"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        c.execute("""SELECT es.status, es.message, es.updated_at, u.name
                     FROM enrollment_status es
                     JOIN users u ON es.user_id = u.user_id
                     WHERE es.user_id = ?""", (user_id,))
        result = c.fetchone()
        conn.close()
        
        if not result:
            return jsonify({"status": "not_found"}), 404
        
        status, message, updated_at, name = result
        
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "name": name,
            "enrollment_status": status,
            "message": message,
            "updated_at": updated_at,
            "completed": status == "success"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/save_template', methods=['POST'])
def save_template():
    """Simpan template final dari ESP32"""
    try:
        data = request.get_json(force=True)
        user_id = data.get('user_id')
        template = data.get('template')
        
        if not user_id or not template:
            return jsonify({"status": "error", "message": "user_id and template required"}), 400
        
        with db_lock:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("SELECT name, status FROM users WHERE user_id=?", (user_id,))
            user = c.fetchone()
            
            if not user:
                conn.close()
                return jsonify({
                    "status": "error", 
                    "message": f"User ID '{user_id}' not found"
                }), 404
            
            user_name, current_status = user
            
            # Save template
            c.execute("""UPDATE users 
                         SET template=?, status='enrolled' 
                         WHERE user_id=?""", 
                      (template, user_id))
            
            # Update enrollment status
            c.execute("""INSERT OR REPLACE INTO enrollment_status 
                         (user_id, status, message, updated_at)
                         VALUES (?, 'success', 'Enrollment completed', ?)""",
                      (user_id, get_wib_time()))
            
            conn.commit()
            conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Template saved successfully",
            "user_id": user_id,
            "name": user_name
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/get_pending_enrollment', methods=['GET'])
def get_pending_enrollment():
    """Get the most recent pending enrollment (for ESP32 auto-detect)"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get the most recent user that is still pending
        c.execute("""SELECT user_id, name 
                     FROM users 
                     WHERE status = 'pending'
                     ORDER BY registered_at DESC 
                     LIMIT 1""")
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({"status": "no_pending"}), 404
        
        return jsonify({
            "status": "success",
            "user_id": user[0],
            "name": user[1]
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ===== VERIFICATION API (from ESP32) =====

@app.route('/api/log_access', methods=['POST'])
def log_access():
    """Log access dari ESP32 (sukses atau gagal)"""
    try:
        data = request.get_json()
        sensor_slot_id = data.get('user_id')
        success = data.get('success', False)
        confidence = data.get('confidence', 0)
        sensor_id = data.get('sensor_id', 0)
        
        with db_lock:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("SELECT user_id FROM sensor_mapping WHERE sensor_slot_id=?", (sensor_slot_id,))
            mapped = c.fetchone()
            
            if mapped:
                user_id = mapped[0]
            else:
                user_id = 'unknown'
            
            name = 'Unknown'
            
            if success and user_id != 'unknown':
                # Get user name
                c.execute("SELECT name FROM users WHERE user_id=?", (user_id,))
                user = c.fetchone()
                
                if user:
                    name = user[0]
                    
                    # Check duplicate
                    is_duplicate, last_time = check_duplicate_access(conn, user_id)
                    if is_duplicate:
                        conn.close()
                        return jsonify({
                            "status": "duplicate",
                            "message": f"Already accessed at {last_time}"
                        })
            
            # Log access (baik sukses maupun gagal)
            c.execute("""INSERT INTO access_logs 
                        (user_id, name, success, confidence, sensor_id, timestamp) 
                        VALUES (?, ?, ?, ?, ?, ?)""", 
                     (user_id, name, 1 if success else 0, confidence, sensor_id, get_wib_time()))
            
            conn.commit()
            conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"Access logged for {name}" if success else "Failed access logged",
            "user_id": user_id,
            "name": name,
            "timestamp": get_wib_time().isoformat()
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ===== SYNC API =====

@app.route('/api/get_all_templates', methods=['GET'])
def get_all_templates():
    """Get all enrolled templates for ESP32 sync"""
    try:
        with db_lock:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT user_id, template, name
                         FROM users
                         WHERE status='enrolled' AND template IS NOT NULL
                         ORDER BY user_id""")
            templates = c.fetchall()
            
            # Clear old mapping
            c.execute("DELETE FROM sensor_mapping")
            
            template_list = []
            sensor_slot_id = 1  # Start dari 1
            for t in templates:
                user_id = t[0]
                template = t[1]
                name = t[2]
                
                # Insert mapping sensor_slot_id → user_id
                c.execute("""INSERT OR REPLACE INTO sensor_mapping 
                             (sensor_slot_id, user_id, synced_at)
                             VALUES (?, ?, ?)""",
                          (sensor_slot_id, user_id, get_wib_time()))
                
                template_list.append({
                    "user_id": user_id,
                    "template": template,
                    "name": name,
                    "sensor_slot_id": sensor_slot_id
                })
                sensor_slot_id += 1
            
            conn.commit()
            conn.close()
        
        return jsonify({
            "status": "success",
            "templates": template_list,
            "total": len(template_list)
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

# ===== DASHBOARD API =====

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    c.execute("""SELECT user_id, name, status, registered_at
                 FROM users ORDER BY registered_at DESC""")
    users = c.fetchall()
    conn.close()
    
    user_list = []
    for user in users:
        user_list.append({
            "user_id": user[0],
            "name": user[1],
            "status": user[2],
            "registered_at": user[3]
        })
    
    return jsonify({"status": "success", "users": user_list})

@app.route('/api/access_logs', methods=['GET'])
def get_access_logs():
    """Get access logs (success & failed)"""
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    c.execute("""SELECT user_id, name, timestamp, success, confidence, sensor_id
                 FROM access_logs 
                 ORDER BY timestamp DESC 
                 LIMIT 100""")
    logs = c.fetchall()
    conn.close()
    
    log_list = []
    for log in logs:
        log_list.append({
            "user_id": log[0],
            "name": log[1],
            "timestamp": log[2],
            "success": bool(log[3]),
            "confidence": log[4],
            "sensor_id": log[5]
        })
    
    return jsonify({"status": "success", "logs": log_list})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics"""
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    # Total users
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]
    
    # Enrolled users
    c.execute("SELECT COUNT(*) FROM users WHERE status='enrolled'")
    enrolled = c.fetchone()[0]
    
    # Today's access (WIB)
    today_start = get_wib_time().replace(hour=0, minute=0, second=0, microsecond=0)
    c.execute("SELECT COUNT(*) FROM access_logs WHERE success=1 AND timestamp >= ?", (today_start,))
    today_success = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM access_logs WHERE success=0 AND timestamp >= ?", (today_start,))
    today_failed = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        "status": "success",
        "stats": {
            "total_users": total_users,
            "enrolled": enrolled,
            "pending": total_users - enrolled,
            "today_success": today_success,
            "today_failed": today_failed,
            "today_total": today_success + today_failed
        }
    })

@app.route('/api/delete_user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete user dan semua datanya"""
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("DELETE FROM enrollment_status WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "User deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/api/debug/mapping', methods=['GET'])
def debug_mapping():
    """Debug endpoint untuk cek sensor mapping"""
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("""SELECT sensor_slot_id, user_id, synced_at 
                     FROM sensor_mapping 
                     ORDER BY sensor_slot_id""")
        mappings = c.fetchall()
        conn.close()
        
        mapping_list = []
        for m in mappings:
            mapping_list.append({
                "sensor_slot_id": m[0],
                "user_id": m[1],
                "synced_at": m[2]
            })
        
        return jsonify({
            "status": "success",
            "mappings": mapping_list,
            "total": len(mapping_list)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)