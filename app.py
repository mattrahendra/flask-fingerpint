from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)
CORS(app)

# Timezone Indonesia (WIB)
WIB = pytz.timezone('Asia/Jakarta')

# Multi-template configuration
REQUIRED_TEMPLATES = 3

# Anti-duplicate check window (dalam detik)
DUPLICATE_CHECK_WINDOW = 10

# Inisialisasi database
def init_db():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    # Tabel users
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    registered_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    templates_count INTEGER DEFAULT 0
                )''')
    
    # Tabel untuk menyimpan multiple templates per user
    c.execute('''CREATE TABLE IF NOT EXISTS user_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    template TEXT NOT NULL,
                    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    template_index INTEGER,
                    sensor_slot_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )''')
    
    # Tabel access logs (sukses dan gagal)
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
    
    # Tabel mapping sensor_slot_id ke user_id (BARU)
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_mapping (
                    sensor_slot_id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    template_db_id INTEGER NOT NULL,
                    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    FOREIGN KEY (template_db_id) REFERENCES user_templates(id) ON DELETE CASCADE
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
        "message": "Server OK - Sensor-based Verification",
        "timestamp": get_wib_time().isoformat(),
        "timezone": "Asia/Jakarta (WIB)",
        "config": {
            "required_templates": REQUIRED_TEMPLATES,
            "duplicate_window": DUPLICATE_CHECK_WINDOW
        }
    })

# ===== ENROLLMENT API =====

@app.route('/api/start_enroll', methods=['POST'])
def start_enroll():
    """Inisialisasi enrollment baru tanpa user_id (auto-generated)"""
    try:
        data = request.get_json()
        name = data.get('name')
        
        if not name:
            return jsonify({"status": "error", "message": "Name is required"}), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        # Generate user_id otomatis (incremental)
        c.execute("SELECT MAX(CAST(user_id AS INTEGER)) FROM users WHERE user_id GLOB '[0-9]*'")
        result = c.fetchone()
        next_id = 1 if not result[0] else int(result[0]) + 1
        user_id = str(next_id).zfill(3)  # Format: 001, 002, 003...
        
        c.execute("""INSERT INTO users (user_id, name, status, registered_at, templates_count) 
                     VALUES (?, ?, 'enrolling', ?, 0)""", 
                  (user_id, name, get_wib_time()))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"Enrollment started for {name}",
            "user_id": user_id,
            "name": name,
            "required_templates": REQUIRED_TEMPLATES
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/save_template', methods=['POST'])
def save_template():
    """Simpan template dari ESP32"""
    try:
        data = request.get_json(force=True)
        user_id = data.get('user_id')
        template = data.get('template')
        
        if not user_id or not template:
            return jsonify({"status": "error", "message": "user_id and template required"}), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("SELECT name, status, templates_count FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            return jsonify({
                "status": "error", 
                "message": f"User ID '{user_id}' not found"
            }), 404
        
        user_name, current_status, templates_count = user
        
        # Simpan template
        new_index = templates_count + 1
        c.execute("""INSERT INTO user_templates 
                     (user_id, template, template_index, enrolled_at) 
                     VALUES (?, ?, ?, ?)""", 
                  (user_id, template, new_index, get_wib_time()))
        
        # Update user status
        new_status = 'enrolled' if new_index >= REQUIRED_TEMPLATES else 'enrolling'
        c.execute("""UPDATE users 
                     SET templates_count=?, status=? 
                     WHERE user_id=?""", 
                  (new_index, new_status, user_id))
        
        conn.commit()
        conn.close()
        
        remaining = REQUIRED_TEMPLATES - new_index
        
        return jsonify({
            "status": "success" if remaining == 0 else "partial",
            "message": f"Template {new_index}/{REQUIRED_TEMPLATES} saved",
            "user_id": user_id,
            "name": user_name,
            "templates_count": new_index,
            "required_templates": REQUIRED_TEMPLATES,
            "remaining": remaining,
            "completed": remaining == 0
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/check_enroll_status/<user_id>', methods=['GET'])
def check_enroll_status(user_id):
    """Check enrollment progress"""
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("SELECT name, status, templates_count FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({"status": "not_found"}), 404
        
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "name": user[0],
            "enrollment_status": user[1],
            "templates_count": user[2],
            "required_templates": REQUIRED_TEMPLATES,
            "completed": user[2] >= REQUIRED_TEMPLATES
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/get_pending_enrollment', methods=['GET'])
def get_pending_enrollment():
    """Get the most recent pending enrollment (for ESP32 auto-detect)"""
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        # Get the most recent user that is still enrolling
        c.execute("""SELECT user_id, name, templates_count 
                     FROM users 
                     WHERE status IN ('pending', 'enrolling')
                     ORDER BY registered_at DESC 
                     LIMIT 1""")
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({"status": "no_pending"}), 404
        
        return jsonify({
            "status": "success",
            "user_id": user[0],
            "name": user[1],
            "templates_count": user[2],
            "required_templates": REQUIRED_TEMPLATES
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ===== VERIFICATION API (from ESP32) =====

@app.route('/api/log_access', methods=['POST'])
def log_access():
    """Log access dari ESP32 (sukses atau gagal)"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'unknown')
        success = data.get('success', False)
        confidence = data.get('confidence', 0)
        sensor_id = data.get('sensor_id', 0)
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        name = "Unknown"
        
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
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("""SELECT ut.id, ut.user_id, ut.template, u.name
                     FROM user_templates ut
                     JOIN users u ON ut.user_id = u.user_id
                     WHERE u.status='enrolled'
                     ORDER BY ut.user_id, ut.template_index""")
        templates = c.fetchall()
        
        # ✅ Clear old mapping
        c.execute("DELETE FROM sensor_mapping")

        template_list = []
        sensor_slot_id = 1
        for t in templates:
            template_db_id = t[0]
            user_id = t[1]
            template = t[2]
            name = t[3]

            # ✅ Insert mapping sensor_slot_id → user_id
            c.execute("""INSERT OR REPLACE INTO sensor_mapping
                        (sensor_slot_id, user_id, template_db_id, synced_at) VALUES (?, ?, ?, ?)""", (sensor_slot_id, user_id, template_db_id, get_wib_time()))

            template_list.append({
                "template_id": template_db_id,
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
    
    c.execute("""SELECT user_id, name, status, registered_at, templates_count
                 FROM users ORDER BY registered_at DESC""")
    users = c.fetchall()
    conn.close()
    
    user_list = []
    for user in users:
        user_list.append({
            "user_id": user[0],
            "name": user[1],
            "status": user[2],
            "registered_at": user[3],
            "templates_count": user[4],
            "required_templates": REQUIRED_TEMPLATES
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
    """Delete user dan semua templatenya"""
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("DELETE FROM user_templates WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "User deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)