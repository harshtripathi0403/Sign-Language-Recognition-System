import cv2
import numpy as np
import joblib
import mediapipe as mp
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import math

class ISLFrontendApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Detection")
        self.root.geometry("1100x650")
        self.root.configure(bg="#0b132b")
        
        # Setup OpenCV and Models
        try:
            self.model = joblib.load("02_repo/isl_model.pkl")
            self.encoder = joblib.load("02_repo/label_encoder.pkl")
        except Exception as e:
            print("Error loading model or encoder. Please ensure you run this from the project root.", e)
            self.model = None
            self.encoder = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.camera_on = True

        # State Variables
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.word_buffer = ""

        self.setup_ui()
        self.update_frame()
        
        # Proper shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # Fonts
        self.title_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.header_font = font.Font(family="Helvetica", size=10, weight="normal")
        self.letter_font = font.Font(family="Helvetica", size=70, weight="bold")
        self.percent_font = font.Font(family="Helvetica", size=30, weight="bold")
        self.word_font = font.Font(family="Courier", size=24, weight="bold")
        self.btn_font = font.Font(family="Helvetica", size=12, weight="bold")

        # Top Bar
        top_bar = tk.Frame(self.root, bg="#111c3a", height=50)
        top_bar.pack(fill=tk.X)
        top_bar.pack_propagate(False)

        title_lbl = tk.Label(top_bar, text="Main Application Window", bg="#111c3a", fg="#4f86f7", font=self.title_font)
        title_lbl.pack(pady=10)

        # Content Area
        content_frame = tk.Frame(self.root, bg="#0b132b")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # -------------------------------------------------------------
        # Left Panel - Video Viewfinder
        # -------------------------------------------------------------
        left_panel = tk.Frame(content_frame, bg="#152042", bd=2, relief=tk.FLAT, highlightbackground="#1d4ed8", highlightthickness=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # We will use a canvas to easily draw the video frames and text overlay
        self.video_canvas = tk.Canvas(left_panel, width=640, height=480, bg="#152042", highlightthickness=0)
        self.video_canvas.pack(expand=True, padx=5, pady=5)
        # Background text for Viewfinder (shown if video is off/loading)
        self.video_canvas.create_text(320, 240, text="Live Viewfinder\n(640 × 480)", fill="#4f86f7", font=("Helvetica", 14), justify=tk.CENTER)

        # Camera control button under viewfinder
        self.btn_toggle_cam = tk.Button(left_panel, text="Stop Camera", bg="#ef4444", fg="white", font=self.btn_font, relief=tk.FLAT, activebackground="#dc2626", activeforeground="white", command=self.toggle_camera)
        self.btn_toggle_cam.pack(pady=10, fill=tk.X, padx=20)

        # -------------------------------------------------------------
        # Right Panel - Controls & Status
        # -------------------------------------------------------------
        right_panel = tk.Frame(content_frame, bg="#0b132b")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, ipadx=10)

        # 1. Prediction Box
        pred_box = tk.Frame(right_panel, bg="#1a2b5e", highlightbackground="#38bdf8", highlightthickness=2)
        pred_box.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(pred_box, text="Predicted Letter", bg="#1a2b5e", fg="#8ea1c1", font=self.header_font, anchor="w").pack(fill=tk.X, padx=10, pady=(5,0))

        pred_inner_frame = tk.Frame(pred_box, bg="#1a2b5e")
        pred_inner_frame.pack(fill=tk.X, padx=10, pady=5)

        self.lbl_letter = tk.Label(pred_inner_frame, text="-", bg="#1a2b5e", fg="#22c55e", font=self.letter_font)
        self.lbl_letter.pack(side=tk.LEFT)

        self.lbl_percent = tk.Label(pred_inner_frame, text="0.0%", bg="#1a2b5e", fg="#22c55e", font=self.percent_font)
        self.lbl_percent.pack(side=tk.RIGHT, pady=(20,0))

        # Progress bar container
        self.prog_canvas = tk.Canvas(pred_box, height=15, bg="#0f172a", highlightthickness=0)
        self.prog_canvas.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.prog_bar = self.prog_canvas.create_rectangle(0, 0, 0, 15, fill="#22c55e", outline="")

        self.lbl_conf_text = tk.Label(pred_box, text="■ Waiting...", bg="#1a2b5e", fg="#22c55e", font=self.header_font, anchor="w")
        self.lbl_conf_text.pack(fill=tk.X, padx=10, pady=(0, 10))

        # 2. Word Buffer Box
        word_box = tk.Frame(right_panel, bg="#1a2b5e", highlightbackground="#2563eb", highlightthickness=2)
        word_box.pack(fill=tk.X, pady=10)

        tk.Label(word_box, text="Word Buffer", bg="#1a2b5e", fg="#8ea1c1", font=self.header_font, anchor="w").pack(fill=tk.X, padx=10, pady=(5,0))
        
        self.lbl_word = tk.Label(word_box, text="_", bg="#1a2b5e", fg="#ffffff", font=self.word_font, anchor="w")
        self.lbl_word.pack(fill=tk.X, padx=10, pady=(10, 20))

        # 3. Action Buttons
        btn_frame = tk.Frame(right_panel, bg="#0b132b")
        btn_frame.pack(fill=tk.X, pady=10)

        btn_commit = tk.Button(btn_frame, text="Commit", bg="#2563eb", fg="white", font=self.btn_font, relief=tk.FLAT, activebackground="#3b82f6", activeforeground="white", command=self.on_commit)
        btn_commit.grid(row=0, column=0, padx=(0, 5), pady=(0, 10), sticky="nsew", ipadx=20, ipady=5)

        btn_space = tk.Button(btn_frame, text="Space", bg="#1e293b", fg="white", font=self.btn_font, relief=tk.FLAT, highlightbackground="#2563eb", highlightthickness=1, activebackground="#334155", activeforeground="white", command=self.on_space)
        btn_space.grid(row=0, column=1, padx=(5, 0), pady=(0, 10), sticky="nsew", ipadx=20, ipady=5)

        btn_delete = tk.Button(btn_frame, text="Delete", bg="#1e293b", fg="white", font=self.btn_font, relief=tk.FLAT, highlightbackground="#2563eb", highlightthickness=1, activebackground="#ef4444", activeforeground="white", command=self.on_delete)
        btn_delete.grid(row=1, column=0, padx=(0, 5), sticky="nsew", ipadx=20, ipady=5)

        btn_clear = tk.Button(btn_frame, text="Clear", bg="#1e293b", fg="white", font=self.btn_font, relief=tk.FLAT, highlightbackground="#2563eb", highlightthickness=1, activebackground="#ef4444", activeforeground="white", command=self.on_clear)
        btn_clear.grid(row=1, column=1, padx=(5, 0), sticky="nsew", ipadx=20, ipady=5)

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.cap.release()
            self.btn_toggle_cam.config(text="Start Camera", bg="#22c55e", activebackground="#16a34a")
            self.clear_video_canvas("Camera Offline")
        else:
            self.cap = cv2.VideoCapture(0)
            self.camera_on = True
            self.btn_toggle_cam.config(text="Stop Camera", bg="#ef4444", activebackground="#dc2626")

    def clear_video_canvas(self, message):
        self.video_canvas.delete("all")
        self.video_canvas.create_text(320, 240, text=message, fill="#4f86f7", font=("Helvetica", 18), justify=tk.CENTER)

    def on_commit(self):
        if self.current_prediction:
            self.word_buffer += self.current_prediction
            self.lbl_word.config(text=self.word_buffer + "_")

    def on_space(self):
        self.word_buffer += " "
        self.lbl_word.config(text=self.word_buffer + "_")

    def on_delete(self):
        if len(self.word_buffer) > 0:
            self.word_buffer = self.word_buffer[:-1]
            self.lbl_word.config(text=self.word_buffer + "_")

    def on_clear(self):
        self.word_buffer = ""
        self.lbl_word.config(text="_")

    def update_frame(self):
        if self.camera_on and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Process Frame
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                
                # Draw overlay and predict
                prediction_made = False
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(img_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        keypoints = []
                        for lm in hand_landmarks.landmark:
                            keypoints.append(lm.x)
                            keypoints.append(lm.y)
                        
                        keypoints = np.array(keypoints)
                        wrist_x = keypoints[0]
                        wrist_y = keypoints[1]
                        
                        for i in range(0, len(keypoints), 2):
                            keypoints[i] -= wrist_x
                            keypoints[i+1] -= wrist_y
                            
                        max_val = np.max(np.abs(keypoints))
                        if max_val != 0:
                            keypoints = keypoints / max_val
                            
                        keypoints = keypoints.reshape(1, -1)
                        
                        if self.model and self.encoder:
                            probs = self.model.predict_proba(keypoints)
                            prediction = self.model.predict(keypoints)
                            label = self.encoder.inverse_transform(prediction)[0]
                            max_prob = np.max(probs) * 100
                            
                            self.current_prediction = label
                            self.current_confidence = max_prob
                            prediction_made = True
                
                # Update UI state
                if prediction_made:
                    self.lbl_letter.config(text=str(self.current_prediction))
                    self.lbl_percent.config(text=f"{self.current_confidence:.1f}%")
                    
                    # Update bar
                    self.prog_canvas.update_idletasks()
                    w = self.prog_canvas.winfo_width()
                    bar_w = int((self.current_confidence / 100.0) * w)
                    
                    # Determine color based on threshold
                    if self.current_confidence > 80:
                        color = "#22c55e" # Green
                        status = "■ High Confidence"
                    elif self.current_confidence > 50:
                        color = "#eab308" # Yellow
                        status = "■ Medium Confidence"
                    else:
                        color = "#ef4444" # Red
                        status = "■ Low Confidence"
                        
                    self.prog_canvas.itemconfig(self.prog_bar, fill=color)
                    self.prog_canvas.coords(self.prog_bar, 0, 0, bar_w, 15)
                    
                    self.lbl_letter.config(fg=color)
                    self.lbl_percent.config(fg=color)
                    self.lbl_conf_text.config(text=status, fg=color)
                else:
                    self.lbl_letter.config(text="-", fg="#8ea1c1")
                    self.lbl_percent.config(text="0.0%", fg="#8ea1c1")
                    self.prog_canvas.coords(self.prog_bar, 0, 0, 0, 15)
                    self.lbl_conf_text.config(text="■ Waiting for hand...", fg="#8ea1c1")
                    self.current_prediction = ""

                # Update Canvas image
                img_pil = Image.fromarray(img_rgb)
                # Resize if necessary to fit the canvas (640x480)
                img_pil = img_pil.resize((640, 480), Image.LANCZOS)
                self.tk_img = ImageTk.PhotoImage(image=img_pil)
                self.video_canvas.delete("all")
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        elif not self.camera_on:
             self.lbl_letter.config(text="-", fg="#8ea1c1")
             self.lbl_percent.config(text="0.0%", fg="#8ea1c1")
             self.prog_canvas.coords(self.prog_bar, 0, 0, 0, 15)
             self.lbl_conf_text.config(text="■ Camera is offline", fg="#8ea1c1")
             self.current_prediction = ""
            
        self.root.after(10, self.update_frame)

    def on_closing(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ISLFrontendApp(root)
    root.mainloop()
