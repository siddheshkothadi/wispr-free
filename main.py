import time
import keyboard
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import sys
import tkinter as tk
from threading import Thread, Lock
import math
import random

# Configuration
MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
HOTKEY = "ctrl+windows"

# Global state for thread communication
class AppState:
    def __init__(self):
        self.is_recording = False
        self.should_show = False
        self.is_processing = False
        self._lock = Lock()
        self._audio_bins = []
        self._audio_peak = 1e-6
        self._audio_level = 0.0

    def update_audio_bins(self, bins):
        if not bins:
            return
        with self._lock:
            self._audio_bins = bins
            peak = max(bins)
            if peak > self._audio_peak:
                self._audio_peak = peak
            else:
                self._audio_peak *= 0.985
                self._audio_peak = max(self._audio_peak, 1e-6)

    def update_audio_level(self, level):
        try:
            level = float(level)
        except Exception:
            return
        with self._lock:
            self._audio_level = 0.85 * self._audio_level + 0.15 * max(0.0, level)

    def reset_audio_bins(self):
        with self._lock:
            self._audio_bins = []
            self._audio_peak = 1e-6
            self._audio_level = 0.0

    def read_audio_bins_normalized(self, n):
        with self._lock:
            bins = list(self._audio_bins)
            peak = float(self._audio_peak)
            level = float(self._audio_level)
        if not bins or peak <= 0:
            return [0.0] * n
        if len(bins) < n:
            bins = bins + [0.0] * (n - len(bins))
        elif len(bins) > n:
            bins = bins[:n]
        # Normalize + boost low levels for better visual response.
        # Also add a touch of overall loudness so speech always shows movement.
        level_boost = min(0.55, level * 18.0)
        out = []
        for b in bins:
            v = min(1.0, max(0.0, b / peak))
            v = math.sqrt(v)  # expand low values
            v = min(1.0, v * 1.6 + level_boost)
            out.append(v)
        return out

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

def compute_audio_bins(indata, n_bins):
    try:
        samples = np.asarray(indata, dtype=np.float32).reshape(-1)
        if samples.size < 64:
            return [0.0] * n_bins

        window = np.hanning(samples.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(samples * window))
        spectrum = spectrum[1:]  # drop DC component
        if spectrum.size == 0:
            return [0.0] * n_bins

        # Use power spectrum with a gentle log to keep speech energy visible.
        spectrum = np.square(spectrum)
        spectrum = np.log1p(spectrum)
        edges = np.linspace(0, spectrum.size, n_bins + 1, dtype=int)
        bins = []
        for i in range(n_bins):
            start = int(edges[i])
            end = int(edges[i + 1])
            if end <= start:
                bins.append(0.0)
                continue
            bins.append(float(np.mean(spectrum[start:end])))
        return bins
    except Exception:
        return [0.0] * n_bins

def record_audio():
    audio_queue = queue.Queue()
    
    def callback(indata, frames, time, status):
        audio_queue.put(indata.copy())
        bins = compute_audio_bins(indata, n_bins=24)
        state.update_audio_bins(bins)
        try:
            samples = np.asarray(indata, dtype=np.float32).reshape(-1)
            rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        except Exception:
            rms = 0.0
        state.update_audio_level(rms)

    state.reset_audio_bins()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback, blocksize=1024):
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
        self.width = 210
        self.height = 44
        self.pill_bg_color = "#000000"
        self.pill_border_color = "#ffffff"
        self.wave_color = "#ffffff"
        self.border_px = 2
        
        # Canvas
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg=self.transparent_color, highlightthickness=0)
        self.canvas.pack()
        
        # Draw the "Pill" shape (black background, minimal white border)
        x1 = self.height // 2
        x2 = self.width - self.height // 2
        y = self.height // 2
        self.pill_border = self.canvas.create_line(
            x1, y, x2, y, fill=self.pill_border_color, width=self.height, capstyle=tk.ROUND
        )
        self.pill_bg = self.canvas.create_line(
            x1, y, x2, y, fill=self.pill_bg_color, width=self.height - (self.border_px * 2), capstyle=tk.ROUND
        )
        
        self.position_window()
        
        # Animation state
        self.phase = 0.0
        self.n_bars = 20
        self.visual_bins = [0.0] * self.n_bars
        self.noise_offsets = [random.uniform(0, math.pi * 2) for _ in range(self.n_bars)]
            
        self.check_state()

    def position_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (self.width // 2)
        y = screen_height - 120
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def _smoothed_bins(self, target_bins, alpha=0.25):
        for i in range(self.n_bars):
            self.visual_bins[i] = (1 - alpha) * self.visual_bins[i] + alpha * target_bins[i]
        return self.visual_bins

    def draw_wave(self):
        if not state.should_show:
            return

        self.canvas.delete("wave")
        
        if state.is_recording:
            center_y = self.height // 2
            bins = state.read_audio_bins_normalized(self.n_bars)

            # Add a subtle motion floor so silence still looks alive
            t = time.time()
            for i in range(self.n_bars):
                bins[i] = max(
                    bins[i],
                    0.07
                    * (0.5 + 0.5 * math.sin(t * 6.0 + self.noise_offsets[i]))
                    * (0.5 + 0.5 * math.sin(t * 3.3 + i * 0.35)),
                )

            bins = self._smoothed_bins(bins, alpha=0.32)

            inner_pad = self.height // 2 + 10
            start_x = inner_pad
            end_x = self.width - inner_pad
            span = max(1, end_x - start_x)
            bar_spacing = span / (self.n_bars - 1)
            bar_width = max(2, int(bar_spacing * 0.42))
            max_bar_height = int(self.height * 0.58)

            for i, v in enumerate(bins):
                envelope = math.sin(math.pi * (i + 1) / (self.n_bars + 1))
                h = int(max_bar_height * v * envelope)
                h = max(2, h)
                x = start_x + int(i * bar_spacing)
                self.canvas.create_line(
                    x,
                    center_y - h // 2,
                    x,
                    center_y + h // 2,
                    tag="wave",
                    fill=self.wave_color,
                    width=bar_width,
                    capstyle=tk.ROUND,
                )
            
        elif state.is_processing:
            # Minimal loading indicator: a straight pulsing line (progress-bar feel)
            center_y = self.height // 2
            inner_pad = self.height // 2 + 10
            min_w = 24
            max_w = self.width - (inner_pad * 2)
            t = time.time()
            w = min_w + (0.5 + 0.5 * math.sin(t * 8.0)) * (max_w - min_w)
            start = (self.width - w) / 2
            end = start + w
            self.canvas.create_line(
                start,
                center_y,
                end,
                center_y,
                tag="wave",
                fill=self.wave_color,
                width=2,
                capstyle=tk.ROUND,
            )
        
        if state.should_show:
            self.root.after(20, self.draw_wave)

    def check_state(self):
        if state.should_show:
            if not self.root.winfo_viewable():
                self.root.deiconify()
                self.phase = 0.0
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
