import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import os

class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Forest Fire Detection System (Genetically Optimized)")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Load the trained model
        try:
            self.model = joblib.load('fire_detection_model.pkl')
            print("Model loaded successfully.")
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file 'fire_detection_model.pkl' not found!\nPlease run the GA training script first.")
            self.root.destroy()
            return

        # --- UI ELEMENTS ---
        
        # Header
        self.header = tk.Label(root, text="Forest Fire Detection System", 
                               font=("Helvetica", 24, "bold"), bg="#2c3e50", fg="white", pady=20)
        self.header.pack(fill=tk.X)

        # Main Content Frame
        self.frame = tk.Frame(root, bg="#f0f0f0")
        self.frame.pack(pady=20)

        # Image Display Area
        self.canvas = tk.Canvas(self.frame, width=400, height=300, bg="#bdc3c7", relief="sunken")
        self.canvas.pack(pady=10)
        
        # Placeholder text
        self.canvas.create_text(200, 150, text="No Image Uploaded", fill="#7f8c8d", font=("Arial", 14))

        # Result Label
        self.result_label = tk.Label(root, text="Status: Waiting for input...", 
                                     font=("Helvetica", 18), bg="#f0f0f0", fg="#34495e")
        self.result_label.pack(pady=10)

        # Buttons Frame
        self.btn_frame = tk.Frame(root, bg="#f0f0f0")
        self.btn_frame.pack(pady=20)

        self.btn_upload = tk.Button(self.btn_frame, text="Upload Image", command=self.upload_image,
                                    font=("Arial", 12), bg="#3498db", fg="white", width=15, height=2)
        self.btn_upload.pack(side=tk.LEFT, padx=20)

        self.btn_detect = tk.Button(self.btn_frame, text="Analyze Image", command=self.detect_fire,
                                    font=("Arial", 12), bg="#e74c3c", fg="white", width=15, height=2, state=tk.DISABLED)
        self.btn_detect.pack(side=tk.LEFT, padx=20)

        self.file_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.file_path = file_path
            
            # Display Image
            img = Image.open(file_path)
            img = img.resize((400, 300), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(200, 150, image=self.photo)
            
            # Reset UI
            self.result_label.config(text="Status: Ready to Analyze", fg="#34495e")
            self.btn_detect.config(state=tk.NORMAL)

    def extract_features(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        img = cv2.resize(img, (128, 128))
        
        # --- NEW LOGIC START ---
        (B, G, R) = cv2.split(img)
        
        # Fire Ratio
        fire_mask = (R > 150) & (R > G) & (R > B)
        fire_ratio = np.sum(fire_mask) / (img.shape[0] * img.shape[1])
        
        # Texture
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])
        
        # Combine exactly like in training: [Fire_Ratio, Mean_R, Mean_G, Mean_B, Std_R, Edge_Density]
        features = [fire_ratio, np.mean(R), np.mean(G), np.mean(B), np.std(R), edge_density]
        # --- NEW LOGIC END ---

        return np.array(features).reshape(1, -1)

    def detect_fire(self):
        if not self.file_path:
            return
            
        features = self.extract_features(self.file_path)
        
        if features is not None:
            # Predict
            # ... inside detect_fire() in app.py ...
            # DEBUGGING: Print what the computer sees
            print("--- DIAGNOSTIC REPORT ---")
            print(f"Fire Ratio (Should be > 0.01): {features[0][0]:.4f}")
            print(f"Red Intensity: {features[0][1]:.2f}")
            
            # Predict
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            if prediction == 1:
                confidence = probability[1] * 100
                self.result_label.config(text=f"WARNING: FIRE DETECTED ({confidence:.2f}%)", fg="#c0392b")
            else:
                confidence = probability[0] * 100
                self.result_label.config(text=f"SAFE: No Fire Detected ({confidence:.2f}%)", fg="#27ae60")
        else:
            messagebox.showerror("Error", "Could not process image.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FireDetectionApp(root)
    root.mainloop()