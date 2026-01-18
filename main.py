import time
import keyboard
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import sys
import tkinter as tk
from threading import Thread
import math
import random

# Configuration
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
HOTKEY = "ctrl+windows"

# Global state for thread communication
class AppState:
    def __init__(self):
        self.is_recording = False
        self.should_show = False
        self.is_processing = False

state = AppState()

def load_model():
    print(f"Loading Whisper model '{MODEL_SIZE}'...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Model loaded.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# Load model globally
model = load_model()

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

def record_audio():
    audio_queue = queue.Queue()
    
    def callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        while keyboard.is_pressed(HOTKEY):
            time.sleep(0.05)
    
    audio_data = []
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())
    
    if not audio_data:
        return np.array([])
    return np.concatenate(audio_data, axis=0)

def transcribe_and_type(audio_data):
    if len(audio_data) == 0:
        return

    audio_data = audio_data.flatten().astype(np.float32)
    segments, info = model.transcribe(audio_data, beam_size=5)
    
    text = ""
    for segment in segments:
        text += segment.text
    
    text = text.strip()
    if text:
        print(f"Transcribed: {text}")
        keyboard.write(text + " ")

def background_listener():
    print(f"Ready! Hold '{HOTKEY}' to record.")
    while True:
        try:
            if keyboard.is_pressed(HOTKEY):
                state.should_show = True
                state.is_recording = True
                state.is_processing = False
                
                audio = record_audio()
                
                state.is_recording = False
                state.is_processing = True
                
                transcribe_and_type(audio)
                
                state.should_show = False
                state.is_processing = False
                
                while keyboard.is_pressed(HOTKEY):
                    time.sleep(0.1)
            time.sleep(0.05)
        except Exception as e:
            print(f"Error: {e}")

class ModernWaveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whispr Overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        
        # Transparent background setup for Windows
        # We use a specific neon color as the "transparent" key
        self.transparent_color = "#ff00ff"
        self.root.wm_attributes("-transparentcolor", self.transparent_color)
        self.root.configure(bg=self.transparent_color)
        
        # Dimensions (Reduced width)
        self.width = 180
        self.height = 50
        
        # Canvas
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg=self.transparent_color, highlightthickness=0)
        self.canvas.pack()
        
        # Draw the "Pill" shape (Black background)
        # Using a thick line with rounded caps acts as a perfect pill
        self.pill_bg = self.canvas.create_line(
            self.height//2, self.height//2, 
            self.width - self.height//2, self.height//2, 
            fill="black", width=self.height, capstyle=tk.ROUND
        )
        
        self.position_window()
        
        # Animation state
        self.phase = 0
        self.waves = []
        colors = ["#00ff99", "#00ccff", "#ff0099"] # Green, Blue, Pink
        for i in range(3):
            self.waves.append({
                "speed": random.uniform(0.2, 0.5),
                "amplitude": random.randint(3, 8),
                "color": colors[i],
                "offset": random.uniform(0, math.pi * 2)
            })
            
        self.check_state()

    def position_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (self.width // 2)
        y = screen_height - 120
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def draw_wave(self):
        if not state.should_show:
            return

        self.canvas.delete("wave")
        
        if state.is_recording:
            center_y = self.height // 2
            
            # Draw multiple overlapping waves
            for i, wave in enumerate(self.waves):
                points = []
                # Keep waves inside the pill padding
                start_x = 20
                end_x = self.width - 20
                
                # Dynamic amplitude
                current_amp = wave["amplitude"] + math.sin(time.time() * 8 + i) * 2
                
                for x in range(start_x, end_x + 1, 4):
                    # Sine wave formula
                    y = center_y + current_amp * math.sin(0.15 * x + self.phase * wave["speed"] + wave["offset"])
                    points.append(x)
                    points.append(y)
                
                self.canvas.create_line(points, tag="wave", fill=wave["color"], width=2, smooth=True)
            
            self.phase += 0.8 # Animation speed
            
        elif state.is_processing:
            # Simple pulsing loading bar
            width_pulse = (math.sin(time.time() * 10) + 1) / 2 * (self.width - 60) + 30
            start = (self.width - width_pulse) / 2
            end = start + width_pulse
            self.canvas.create_line(start, self.height//2, end, self.height//2, tag="wave", fill="#00ccff", width=4, capstyle=tk.ROUND)
        
        if state.should_show:
            self.root.after(20, self.draw_wave)

    def check_state(self):
        if state.should_show:
            if not self.root.winfo_viewable():
                self.root.deiconify()
                self.phase = 0
                self.draw_wave()
        else:
            if self.root.winfo_viewable():
                self.root.withdraw()

        self.root.after(50, self.check_state)

def main():
    listener_thread = Thread(target=background_listener, daemon=True)
    listener_thread.start()
    
    root = tk.Tk()
    app = ModernWaveApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
