from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inisialisasi database
def init_db():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    # Tabel users (data dari web)
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    template TEXT,
                    registered_at TIMESTAMP,
                    enrolled_at TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                )''')
    
    # Tabel attendance logs - tambah kolom confidence
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    name TEXT,
                    confidence INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )''')
    
    conn.commit()
    conn.close()

init_db()

# === WEB ROUTES ===
@app.route('/')
def home():
    return render_template('index.html')

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
            "server": "Flask",
            "timestamp": datetime.now().isoformat()
        })

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    # Get all users
    c.execute("SELECT user_id, name, status, registered_at, enrolled_at FROM users ORDER BY registered_at DESC")
    users = c.fetchall()
    
    # Get recent attendance
    c.execute("""SELECT user_id, name, confidence, timestamp 
                 FROM attendance 
                 ORDER BY timestamp DESC 
                 LIMIT 10""")
    attendance = c.fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', users=users, attendance=attendance)

# === API ENDPOINTS ===

# 1. Registrasi user dari web (Point A)
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
        
        # Cek apakah user_id sudah ada
        c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if c.fetchone():
            conn.close()
            return jsonify({"status": "error", "message": "User ID sudah terdaftar"}), 400
        
        # Insert user baru dengan status pending
        c.execute("""INSERT INTO users (user_id, name, status, registered_at) 
                     VALUES (?, ?, 'pending', ?)""", 
                  (user_id, name, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"User {name} berhasil didaftarkan. Silakan scan fingerprint di ESP32 Sensor 1.",
            "user_id": user_id
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 2. Cek status user (untuk polling dari web)
@app.route('/api/check_status/<user_id>', methods=['GET'])
def check_status(user_id):
    try:
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("SELECT status FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({"status": "not_found"}), 404
        
        return jsonify({"status": user[0]})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 3. Simpan template dari ESP32 SENSOR 1 (Point B)
@app.route('/api/save_template', methods=['POST'])
def save_template():
    try:
        if not request.data:
            return jsonify({
                "status": "error", 
                "message": "No data received"
            }), 400
        
        raw_data = request.get_data(as_text=True)
        print(f"[DEBUG] Raw data: {raw_data[:200]}")
        
        try:
            data = request.get_json(force=True)
        except Exception as json_error:
            print(f"[ERROR] JSON parse error: {str(json_error)}")
            return jsonify({
                "status": "error", 
                "message": f"Invalid JSON: {str(json_error)}"
            }), 400
        
        if not data:
            return jsonify({
                "status": "error", 
                "message": "Empty JSON data"
            }), 400
        
        print(f"[DEBUG] JSON keys: {list(data.keys())}")
        
        user_id = data.get('user_id') or data.get('id') or data.get('userId')
        template = data.get('template') or data.get('fingerprint_template')
        
        print(f"[DEBUG] Extracted - user_id: '{user_id}', template length: {len(template) if template else 0}")
        
        if not user_id:
            available_keys = ', '.join(data.keys())
            return jsonify({
                "status": "error", 
                "message": f"user_id wajib diisi. Available keys: {available_keys}"
            }), 400
            
        if not template:
            return jsonify({
                "status": "error", 
                "message": "template wajib diisi"
            }), 400
        
        if len(template) < 100:
            return jsonify({
                "status": "error",
                "message": f"Template terlalu pendek ({len(template)} chars). Expected ~1024 chars"
            }), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        c.execute("SELECT name, status FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            return jsonify({
                "status": "error", 
                "message": f"User ID '{user_id}' tidak ditemukan. Daftar dulu di web!"
            }), 404
        
        user_name, current_status = user
        
        if current_status == 'enrolled':
            conn.close()
            return jsonify({
                "status": "warning",
                "message": f"User {user_name} sudah enrolled sebelumnya"
            }), 200
        
        c.execute("""UPDATE users 
                     SET template=?, status='enrolled', enrolled_at=? 
                     WHERE user_id=?""", 
                  (template, datetime.now(), user_id))
        
        affected = c.rowcount
        conn.commit()
        conn.close()
        
        print(f"[SUCCESS] Updated {affected} row(s) for user_id: {user_id}")
        
        return jsonify({
            "status": "success",
            "message": f"Template untuk {user_name} berhasil disimpan!",
            "user_id": user_id,
            "name": user_name
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": f"Server error: {str(e)}"
        }), 500

# 4. Get all users dengan template (untuk ESP32 SENSOR 2)
@app.route('/api/users', methods=['GET'])
def get_users():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    c.execute("SELECT user_id, name, status, registered_at, enrolled_at, template FROM users")
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
            "template": user[5] if user[5] else ""
        })
    
    return jsonify({"status": "success", "users": user_list})

# 5. Record attendance dari ESP32 SENSOR 2
@app.route('/api/record_attendance', methods=['POST'])
def record_attendance():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        name = data.get('name')
        confidence = data.get('confidence', 0)
        
        if not user_id or not name:
            return jsonify({"status": "error", "message": "user_id dan name wajib diisi"}), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        # Catat attendance
        c.execute("INSERT INTO attendance (user_id, name, confidence) VALUES (?, ?, ?)", 
                 (user_id, name, confidence))
        conn.commit()
        conn.close()
        
        print(f"[ATTENDANCE] {name} ({user_id}) - Confidence: {confidence}")
        
        return jsonify({
            "status": "success",
            "message": f"✅ Attendance recorded for {name}!",
            "user_id": user_id,
            "name": name,
            "confidence": confidence
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 6. Get attendance logs
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    c.execute("""SELECT user_id, name, confidence, timestamp 
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
            "confidence": log[2],
            "timestamp": log[3]
        })
    
    return jsonify({"status": "success", "attendance": attendance_list})

# 7. Delete user (untuk cancel enrollment)
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
# 8. Upload fingerprint image (dari ESP32)
@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # Tentukan nama file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"finger_{timestamp}.bmp"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Simpan file
        file.save(save_path)

        print(f"[UPLOAD] Image saved to {save_path}")

        # Kembalikan URL publik
        file_url = f"/uploads/{filename}"

        return jsonify({
            "status": "success",
            "message": "Image uploaded successfully",
            "url": file_url
        })

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# 9. Serve uploaded images
@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
