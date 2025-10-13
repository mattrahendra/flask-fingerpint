from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

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
    
    # Tabel attendance logs
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    name TEXT,
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

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    
    # Get all users
    c.execute("SELECT user_id, name, status, registered_at, enrolled_at FROM users ORDER BY registered_at DESC")
    users = c.fetchall()
    
    # Get recent attendance
    c.execute("""SELECT user_id, name, timestamp 
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
            "message": f"User {name} berhasil didaftarkan. Silakan scan fingerprint di ESP32.",
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

# 3. Simpan template dari ESP32 (Point B)
@app.route('/api/save_template', methods=['POST'])
def save_template():
    try:
        # Debug: Print raw data
        raw_data = request.get_data(as_text=True)
        print(f"[DEBUG] Raw data received: {raw_data[:100]}...")  # Print first 100 chars
        
        data = request.get_json()
        print(f"[DEBUG] Parsed JSON keys: {list(data.keys()) if data else 'None'}")
        
        # Validasi input - bisa "user_id" atau "id"
        user_id = data.get('user_id') or data.get('id')
        template = data.get('template') or data.get('fingerprint_template')
        
        print(f"[DEBUG] user_id: {user_id}, template length: {len(template) if template else 0}")
        
        if not user_id:
            return jsonify({
                "status": "error", 
                "message": "user_id wajib diisi"
            }), 400
            
        if not template:
            return jsonify({
                "status": "error", 
                "message": "template wajib diisi"
            }), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        # Cek apakah user ada
        c.execute("SELECT name, status FROM users WHERE user_id=?", (user_id,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            return jsonify({
                "status": "error", 
                "message": f"User ID {user_id} tidak ditemukan. Daftar dulu di web!"
            }), 404
        
        user_name, current_status = user
        
        # Cek apakah sudah enrolled sebelumnya
        if current_status == 'enrolled':
            conn.close()
            return jsonify({
                "status": "warning",
                "message": f"User {user_name} sudah enrolled sebelumnya"
            })
        
        # Update template dan status
        c.execute("""UPDATE users 
                     SET template=?, status='enrolled', enrolled_at=? 
                     WHERE user_id=?""", 
                  (template, datetime.now(), user_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": f"✅ Template untuk {user_name} berhasil disimpan!",
            "user_id": user_id,
            "name": user_name
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Server error: {str(e)}"
        }), 500

# 3. Verifikasi fingerprint dari ESP32
@app.route('/api/verify_template', methods=['POST'])
def verify_template():
    try:
        data = request.get_json()
        new_template = data.get('template')
        
        if not new_template:
            return jsonify({"status": "error", "message": "template wajib diisi"}), 400
        
        conn = sqlite3.connect('fingerprint.db')
        c = conn.cursor()
        
        # Ambil semua user yang sudah enrolled
        c.execute("SELECT user_id, name, template FROM users WHERE status='enrolled'")
        users = c.fetchall()
        
        if not users:
            conn.close()
            return jsonify({"status": "no_match", "message": "Belum ada user yang terdaftar"})
        
        # Bandingkan template (simple matching)
        best_match = None
        best_similarity = 0
        
        for user in users:
            user_id, name, stored_template = user
            
            # Hitung similarity (persentase karakter yang sama)
            if len(stored_template) == len(new_template):
                matches = sum(a == b for a, b in zip(stored_template, new_template))
                similarity = matches / len(stored_template)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (user_id, name)
        
        # Threshold 85% similarity
        if best_match and best_similarity > 0.85:
            user_id, name = best_match
            
            # Catat attendance
            c.execute("INSERT INTO attendance (user_id, name) VALUES (?, ?)", 
                     (user_id, name))
            conn.commit()
            conn.close()
            
            return jsonify({
                "status": "match",
                "user_id": user_id,
                "name": name,
                "similarity": round(best_similarity * 100, 2),
                "message": f"✅ Selamat datang, {name}!"
            })
        else:
            conn.close()
            return jsonify({
                "status": "no_match",
                "message": "❌ Sidik jari tidak terdaftar",
                "best_similarity": round(best_similarity * 100, 2) if best_similarity > 0 else 0
            })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 4. Delete user (untuk cancel enrollment)
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

# 5. Get all users (untuk dashboard)
@app.route('/api/users', methods=['GET'])
def get_users():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    c.execute("SELECT user_id, name, status, registered_at, enrolled_at FROM users")
    users = c.fetchall()
    conn.close()
    
    user_list = []
    for user in users:
        user_list.append({
            "user_id": user[0],
            "name": user[1],
            "status": user[2],
            "registered_at": user[3],
            "enrolled_at": user[4]
        })
    
    return jsonify({"status": "success", "users": user_list})

# 5. Get attendance logs
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    conn = sqlite3.connect('fingerprint.db')
    c = conn.cursor()
    c.execute("""SELECT user_id, name, timestamp 
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
            "timestamp": log[2]
        })
    
    return jsonify({"status": "success", "attendance": attendance_list})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)