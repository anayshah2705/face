import os
import datetime
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import base64

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")  # Replace with your JSON file path
firebase_admin.initialize_app(cred)
db = firestore.client()


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")
        self.main_window.title("Facial Recognition Attendance System")

        # Buttons
        self.login_button = tk.Button(self.main_window, text="Login", bg="green", fg="white", command=self.login)
        self.login_button.place(x=750, y=200)

        self.logout_button = tk.Button(self.main_window, text="Logout", bg="red", fg="white", command=self.logout)
        self.logout_button.place(x=750, y=300)

        self.register_button = tk.Button(self.main_window, text="Register New User", bg="gray", fg="black",
                                         command=self.register_new_user)
        self.register_button.place(x=750, y=400)

        # Webcam Label
        self.webcam_label = tk.Label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture_arr = frame
            img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    def recognize(self, frame):
        # Get all known users from Firestore
        users_ref = db.collection("users")
        users = users_ref.stream()

        known_face_encodings = []
        known_face_names = []

        for user in users:
            user_data = user.to_dict()
            name = user_data["name"]
            # Decode base64 string back to numpy array
            encoding_str = user_data["encoding"]
            encoding_bytes = base64.b64decode(encoding_str)
            encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
            known_face_encodings.append(encoding)
            known_face_names.append(name)

        # Recognize face in the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        if not face_encodings:
            return "no_persons_found"

        face_encoding = face_encodings[0]  # Assume one face for simplicity
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

        if True in matches:
            first_match_index = matches.index(True)
            return known_face_names[first_match_index]
        return "unknown_person"

    def login(self):
        name = self.recognize(self.most_recent_capture_arr)

        if name in ["unknown_person", "no_persons_found"]:
            tk.messagebox.showerror("Error", "Unknown user. Please register or try again.")
        else:
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            attendance_ref = db.collection("attendance").document(f"{name}_{date_str}")
            doc = attendance_ref.get()

            if not doc.exists:
                attendance_ref.set({
                    "name": name,
                    "date": date_str,
                    "time_in": time_str,
                    "time_out": None
                })
                tk.messagebox.showinfo("Welcome", f"Welcome, {name}. Logged in at {time_str}.")
            else:
                tk.messagebox.showinfo("Already Logged In", f"{name}, you are already logged in today.")

    def logout(self):
        name = self.recognize(self.most_recent_capture_arr)

        if name in ["unknown_person", "no_persons_found"]:
            tk.messagebox.showerror("Error", "Unknown user. Please register or try again.")
        else:
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            attendance_ref = db.collection("attendance").document(f"{name}_{date_str}")
            doc = attendance_ref.get()

            if doc.exists and not doc.to_dict().get("time_out"):
                attendance_ref.update({"time_out": time_str})
                tk.messagebox.showinfo("Goodbye", f"Goodbye, {name}. Logged out at {time_str}.")
            else:
                tk.messagebox.showinfo("Not Logged In", f"{name}, you haven’t logged in today or already logged out.")

    def register_new_user(self):
        self.register_window = tk.Toplevel(self.main_window)
        self.register_window.geometry("1200x520+370+120")
        self.register_window.title("Register New User")

        # Accept and Try Again Buttons
        self.accept_button = tk.Button(self.register_window, text="Accept", bg="green", fg="white",
                                       command=self.accept_register_new_user)
        self.accept_button.place(x=750, y=300)

        self.try_again_button = tk.Button(self.register_window, text="Try Again", bg="red", fg="white",
                                          command=self.try_again_register_new_user)
        self.try_again_button.place(x=750, y=400)

        # Capture Label
        self.capture_label = tk.Label(self.register_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)
        self.add_img_to_label(self.capture_label)

        # Username Entry
        self.entry_text = tk.Text(self.register_window, height=1, width=20)
        self.entry_text.place(x=750, y=150)

        tk.Label(self.register_window, text="Please input username:").place(x=750, y=70)

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text.get("1.0", "end-1c").strip()
        if not name:
            tk.messagebox.showerror("Error", "Please enter a username.")
            return

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)
        if not embeddings:
            tk.messagebox.showerror("Error", "No face detected. Please try again.")
            return

        # Convert embedding to base64 string for Firestore storage
        embedding = embeddings[0]
        embedding_bytes = embedding.tobytes()
        embedding_str = base64.b64encode(embedding_bytes).decode("utf-8")

        # Store in Firestore
        db.collection("users").document(name).set({
            "name": name,
            "encoding": embedding_str
        })

        tk.messagebox.showinfo("Success", "User registered successfully!")
        self.register_window.destroy()

    def try_again_register_new_user(self):
        self.register_window.destroy()

    def start(self):
        self.main_window.mainloop()

    def __del__(self):
        if 'cap' in self.__dict__:
            self.cap.release()


if __name__ == "__main__":
    app = App()
    app.start()