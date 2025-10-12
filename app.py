from flask import Flask, request, jsonify, render_template
import sqlite3

app = Flask(__name__)

# --- Inisialisasi database ---
def init_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    template TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# --- Halaman utama (web form Point A) ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Endpoint untuk menyimpan fingerprint (enroll) ---
@app.route('/enroll', methods=['POST'])
def enroll():
    data = request.get_json()
    user_id = data.get('id')
    name = data.get('name')
    template = data.get('fingerprint_template')
    
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    if c.fetchone():
        conn.close()
        return jsonify({"status": "error", "message": "ID sudah ada"}), 400
    
    c.execute("INSERT INTO users (id, name, template) VALUES (?, ?, ?)", 
              (user_id, name, template))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success", "message": f"User {name} berhasil disimpan."})

# --- Endpoint untuk verifikasi (Point B) ---
@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    new_template = data.get('fingerprint_template')
    
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT id, name, template FROM users")
    users = c.fetchall()
    conn.close()
    
    for user in users:
        stored_template = user[2]
        same = sum(a == b for a, b in zip(stored_template, new_template))
        similarity = same / max(len(stored_template), len(new_template))
        
        if similarity > 0.9:
            return jsonify({"status": "match", "id": user[0], "name": user[1]})
    
    return jsonify({"status": "no_match"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)