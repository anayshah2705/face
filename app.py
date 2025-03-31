from flask import Flask, render_template, request, jsonify
import cv2
import os
import pickle
import numpy as np
import face_recognition
import base64
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
db_path = "db"
log_path = "log.txt"
os.makedirs(db_path, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def save_log(name, status):
    with open(log_path, 'a') as f:
        f.write(f"{name},{datetime.now()},{status}\n")


def recognize_face(img_data):
    try:
        # Decode base64 image
        img_array = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Convert to RGB (required for face_recognition)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)

        if not encodings:
            return "no_persons_found"

        encoding_unknown = encodings[0]
        for file in os.listdir(db_path):
            with open(os.path.join(db_path, file), 'rb') as f:
                known_encoding = pickle.load(f)
            if face_recognition.compare_faces([known_encoding], encoding_unknown)[0]:
                return file.replace(".pickle", "")
        return "unknown_person"
    except Exception as e:
        print("Face recognition error:", str(e))
        return "error"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/action', methods=['POST'])
def action():
    data = request.json
    action_type = data.get('action')
    img_data = data.get('image')

    if not img_data:
        return jsonify({"error": "No image received"}), 400

    if action_type == 'register':
        name = data.get('name', '').strip()
        if not name:
            return jsonify({"error": "Missing name"}), 400

        # Decode image and extract face encoding
        img_array = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)

        if not encodings:
            return jsonify({"error": "No face detected"}), 400

        # Save face encoding
        with open(os.path.join(db_path, f"{name}.pickle"), 'wb') as f:
            pickle.dump(encodings[0], f)

        # Save image for reference
        img_path = os.path.join(UPLOAD_FOLDER, f"{name}.jpg")
        cv2.imwrite(img_path, img)

        return jsonify({"message": "User registered successfully!", "image_path": img_path})

    elif action_type in ['checkin', 'checkout']:
        name = recognize_face(img_data)

        if name in ["unknown_person", "no_persons_found", "error"]:
            return jsonify({"error": "User not recognized"}), 401

        status = "in" if action_type == "checkin" else "out"
        save_log(name, status)
        return jsonify({"message": f"{'Welcome' if action_type == 'checkin' else 'Goodbye'}, {name}!"})

    return jsonify({"error": "Invalid action type"}), 400


if __name__ == '__main__':
    app.run(debug=True)
