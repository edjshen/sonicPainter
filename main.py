import streamlit as st
import numpy as np
import cv2
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import os
from scipy import signal
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
from collections import deque

def main():
    st.set_page_config(
        page_title="shitty visualizer ejs 2024",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for TouchDesigner-like styling
    st.markdown("""
    <style>
    .main-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .node-container {
        background: #2d2d2d;
        border: 2px solid #4a9eff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(74, 158, 255, 0.3);
    }
    
    .connection-line {
        border-top: 2px solid #4a9eff;
        margin: 20px 0;
        position: relative;
    }
    
    .connection-line::before {
        content: "▶";
        position: absolute;
        right: -10px;
        top: -10px;
        color: #4a9eff;
        font-size: 20px;
    }
    
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #4a9eff, #00ff88);
    }
    
    h1, h2, h3 {
        color: #4a9eff;
        text-shadow: 0 0 10px rgba(74, 158, 255, 0.5);
    }
    
    .metric-container {
        background: rgba(74, 158, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4a9eff;
    }
    
    .oscilloscope-info {
        background: rgba(255, 165, 0, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffa500;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎵 yay! visualizer!")
    st.markdown("*Real-time audio-reactive visual generation with oscilloscope oooooh aaaaaahhhhhh*")
    
    # Initialize session state
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'sr' not in st.session_state:
        st.session_state.sr = None
    if 'processed_frames' not in st.session_state:
        st.session_state.processed_frames = []
    if 'stereo_audio' not in st.session_state:
        st.session_state.stereo_audio = None

    # Main layout with columns for node-based interface
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("### 🎤 Audio Input Node")
        st.markdown('<div class="node-container">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Audio File", 
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload your audio file for visualization. For oscilloscope mode, stereo files work best!"
        )
        
        if uploaded_file:
            # Load audio - keep stereo for oscilloscope mode
            try:
                # Load as stereo first
                stereo_audio, sr = librosa.load(uploaded_file, sr=22050, mono=False)
                
                # Handle mono vs stereo
                if stereo_audio.ndim == 1:
                    # Mono file - duplicate for stereo
                    st.session_state.stereo_audio = np.array([stereo_audio, stereo_audio])
                    st.session_state.audio_data = stereo_audio
                    st.warning("📻 Mono audio detected - duplicated to stereo for oscilloscope mode")
                else:
                    # Stereo file
                    st.session_state.stereo_audio = stereo_audio
                    st.session_state.audio_data = np.mean(stereo_audio, axis=0)  # Mix to mono for other modes
                    st.success("🎧 Stereo audio detected - perfect for oscilloscope mode!")
                    
                st.session_state.sr = sr
                
            except Exception as e:
                st.error(f"Error loading audio: {e}")
                
            if st.session_state.audio_data is not None:
                # Audio info
                duration = len(st.session_state.audio_data) / sr
                channels = "Stereo" if st.session_state.stereo_audio.ndim == 2 else "Mono"
                
                st.markdown(f"**Duration:** {duration:.2f}s")
                st.markdown(f"**Sample Rate:** {sr} Hz")
                st.markdown(f"**Channels:** {channels}")
                
                # Play audio
                st.audio(uploaded_file)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Audio Analysis Node
        if st.session_state.audio_data is not None:
            st.markdown("### 📊 Audio Analysis Node")
            st.markdown('<div class="node-container">', unsafe_allow_html=True)
            
            # Beat tracking - tempo is returned as array
            tempo, beats = librosa.beat.beat_track(
                y=st.session_state.audio_data, 
                sr=st.session_state.sr
            )
            
            # Extract scalar values from arrays
            tempo_scalar = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            
            st.markdown(f"**Tempo:** {tempo_scalar:.1f} BPM")
            st.markdown(f"**Beats Detected:** {len(beats)}")
            
            # Spectral features - these also return arrays
            spectral_centroids = librosa.feature.spectral_centroid(
                y=st.session_state.audio_data, 
                sr=st.session_state.sr
            )[0]
            
            # Get mean spectral centroid
            mean_centroid = float(np.mean(spectral_centroids))
            
            mfccs = librosa.feature.mfcc(
                y=st.session_state.audio_data, 
                sr=st.session_state.sr
            )
            
            # Display additional audio features
            st.markdown(f"**Spectral Centroid:** {mean_centroid:.1f} Hz")
            st.markdown(f"**MFCC Features:** {mfccs.shape[0]} coefficients")
            
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎨 Visual Generator Nodes")
        
        # Connection line
        st.markdown('<div class="connection-line"></div>', unsafe_allow_html=True)
        
        # Visual Parameters
        st.markdown('<div class="node-container">', unsafe_allow_html=True)
        
        visual_type = st.selectbox(
            "Visual Type",
            ["Circular Spectrum", "Waveform Tunnel", "Fractal Mandala", "Audio Reactive Mesh", "Oscilloscope XY", "Oscilloscope Lines"]
        )
        
        # Show oscilloscope info
        if visual_type in ["Oscilloscope XY", "Oscilloscope Lines"]:
            st.markdown("""
            <div class="oscilloscope-info">
            <strong>🔬 Oscilloscope Mode</strong><br>
            • <strong>XY Mode:</strong> Left channel = X, Right channel = Y coordinates<br>
            • <strong>Lines Mode:</strong> Draws connected waveform paths<br>
            • Best with stereo audio or oscilloscope music!
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time parameters
        col_a, col_b = st.columns(2)
        with col_a:
            intensity = st.slider("Intensity", 0.1, 5.0, 1.0, 0.1)
            color_mode = st.selectbox("Color Mode", ["Spectrum", "Monochrome", "Heat", "Oscilloscope Green"])
            
        with col_b:
            reactivity = st.slider("Audio Reactivity", 0.0, 2.0, 1.0, 0.1)
            smoothing = st.slider("Smoothing", 0.0, 1.0, 0.3, 0.05)
        
        # Oscilloscope-specific parameters
        if visual_type in ["Oscilloscope XY", "Oscilloscope Lines"]:
            st.markdown("**🔬 Oscilloscope Settings**")
            col_c, col_d = st.columns(2)
            with col_c:
                trace_length = st.slider("Trace Length", 100, 2000, 500, 50)
                line_width = st.slider("Line Width", 1, 5, 2, 1)
            with col_d:
                persistence = st.slider("Persistence", 0.0, 0.9, 0.3, 0.05)
                zoom = st.slider("Zoom", 0.1, 3.0, 1.0, 0.1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate Visuals
        if st.session_state.audio_data is not None:
            if st.button("🎬 Generate Visualization", type="primary"):
                with st.spinner("Generating audio-reactive visuals..."):
                    # Pass additional parameters for oscilloscope modes
                    extra_params = {}
                    if visual_type in ["Oscilloscope XY", "Oscilloscope Lines"]:
                        extra_params = {
                            'trace_length': trace_length,
                            'line_width': line_width,
                            'persistence': persistence,
                            'zoom': zoom
                        }
                    
                    video_frames = generate_visualization(
                        st.session_state.audio_data,
                        st.session_state.sr,
                        visual_type,
                        intensity,
                        reactivity,
                        color_mode,
                        smoothing,
                        st.session_state.stereo_audio,
                        **extra_params
                    )
                    st.session_state.processed_frames = video_frames
                    st.success("Visualization generated!")
            
            # Preview
            if st.session_state.processed_frames:
                st.markdown("### 👁️ Preview")
                frame_idx = st.slider("Frame", 0, len(st.session_state.processed_frames)-1, 0)
                st.image(st.session_state.processed_frames[frame_idx], use_column_width=True)
    
    with col3:
        st.markdown("### 🎬 Output Node")
        st.markdown('<div class="node-container">', unsafe_allow_html=True)
        
        # Export settings
        fps = st.selectbox("Frame Rate", [24, 30, 60], index=1)
        resolution = st.selectbox("Resolution", ["720p", "1080p"], index=0)
        
        # Export video
        if st.session_state.processed_frames and st.button("📹 Export Video"):
            with st.spinner("Exporting video..."):
                video_path = export_video(
                    st.session_state.processed_frames,
                    st.session_state.audio_data,
                    st.session_state.sr,
                    fps,
                    resolution
                )
                
                # Download link
                with open(video_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Video",
                        data=file.read(),
                        file_name="audio_visualization.mp4",
                        mime="video/mp4"
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance metrics
        if st.session_state.processed_frames:
            st.markdown("### 📈 Performance")
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Frames Generated", len(st.session_state.processed_frames))
            st.metric("Memory Usage", f"{len(st.session_state.processed_frames) * 0.5:.1f} MB")
            st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data
def generate_visualization(audio_data, sr, visual_type, intensity, reactivity, color_mode, smoothing, stereo_audio=None, **kwargs):
    """Generate visualization frames based on audio analysis"""
    
    # Audio analysis
    hop_length = 512
    n_frames = len(audio_data) // hop_length
    
    # Extract features
    stft = librosa.stft(audio_data, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Frequency bands for reactivity
    freqs = librosa.fft_frequencies(sr=sr)
    low_freq_idx = np.where(freqs <= 200)[0]
    mid_freq_idx = np.where((freqs > 200) & (freqs <= 2000))[0]
    high_freq_idx = np.where(freqs > 2000)[0]
    
    frames = []
    prev_frame = None
    
    # Initialize oscilloscope trace buffer
    if visual_type in ["Oscilloscope XY", "Oscilloscope Lines"]:
        trace_buffer = deque(maxlen=kwargs.get('trace_length', 500))
    
    for frame_idx in range(min(n_frames, 300)):  # Limit frames for demo
        # Extract frequency band energies
        low_energy = np.mean(magnitude[low_freq_idx, frame_idx]) * reactivity
        mid_energy = np.mean(magnitude[mid_freq_idx, frame_idx]) * reactivity
        high_energy = np.mean(magnitude[high_freq_idx, frame_idx]) * reactivity
        
        # Generate frame based on visual type
        if visual_type == "Circular Spectrum":
            frame = generate_circular_spectrum(
                magnitude[:, frame_idx], 
                intensity, 
                color_mode,
                low_energy,
                mid_energy,
                high_energy
            )
        elif visual_type == "Waveform Tunnel":
            frame = generate_waveform_tunnel(
                audio_data[frame_idx*hop_length:(frame_idx+1)*hop_length],
                intensity,
                color_mode,
                low_energy,
                frame_idx
            )
        elif visual_type == "Fractal Mandala":
            frame = generate_fractal_mandala(
                low_energy,
                mid_energy,
                high_energy,
                intensity,
                color_mode,
                frame_idx
            )
        elif visual_type == "Oscilloscope XY":
            frame = generate_oscilloscope_xy(
                stereo_audio,
                frame_idx,
                hop_length,
                intensity,
                color_mode,
                trace_buffer,
                **kwargs
            )
        elif visual_type == "Oscilloscope Lines":
            frame = generate_oscilloscope_lines(
                stereo_audio,
                frame_idx,
                hop_length,
                intensity,
                color_mode,
                trace_buffer,
                **kwargs
            )
        else:  # Audio Reactive Mesh
            frame = generate_audio_mesh(
                magnitude[:, frame_idx],
                intensity,
                color_mode,
                frame_idx
            )
        
        # Apply smoothing
        if prev_frame is not None and smoothing > 0:
            frame = cv2.addWeighted(prev_frame, smoothing, frame, 1-smoothing, 0)
        
        frames.append(frame)
        prev_frame = frame.copy()
    
    return frames

def generate_oscilloscope_xy(stereo_audio, frame_idx, hop_length, intensity, color_mode, trace_buffer, **kwargs):
    """Generate oscilloscope XY mode visualization - left=X, right=Y"""
    size = 720
    center = size // 2
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    if stereo_audio is None or stereo_audio.ndim != 2:
        return img
    
    # Extract audio chunk for this frame
    start_sample = frame_idx * hop_length
    end_sample = start_sample + hop_length
    
    if end_sample >= stereo_audio.shape[1]:
        return img
    
    left_channel = stereo_audio[0, start_sample:end_sample]
    right_channel = stereo_audio[1, start_sample:end_sample]
    
    # Get oscilloscope parameters
    zoom = kwargs.get('zoom', 1.0)
    line_width = kwargs.get('line_width', 2)
    persistence = kwargs.get('persistence', 0.3)
    
    # Apply persistence (fade previous frame)
    if persistence > 0:
        img = (img * persistence).astype(np.uint8)
    
    # Scale audio to screen coordinates
    scale = (size // 2 - 50) * intensity * zoom
    
    # Convert audio samples to screen coordinates
    for i in range(len(left_channel)):
        x = int(center + left_channel[i] * scale)
        y = int(center + right_channel[i] * scale)
        
        # Clamp to screen bounds
        x = max(0, min(size-1, x))
        y = max(0, min(size-1, y))
        
        # Add to trace buffer
        trace_buffer.append((x, y))
    
    # Draw the trace
    if len(trace_buffer) > 1:
        points = list(trace_buffer)
        
        for i in range(1, len(points)):
            x1, y1 = points[i-1]
            x2, y2 = points[i]
            
            # Color based on position in trace (newer = brighter)
            alpha = i / len(points)
            
            if color_mode == "Oscilloscope Green":
                color = (0, int(255 * alpha), 0)
            elif color_mode == "Spectrum":
                hue = (frame_idx + i) % 360
                color = hsv_to_rgb(hue, 1.0, alpha)
            elif color_mode == "Heat":
                color = (int(255 * alpha), int(128 * alpha), 0)
            else:  # Monochrome
                val = int(255 * alpha)
                color = (val, val, val)
            
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)
    
    # Draw center crosshairs
    cv2.line(img, (center-20, center), (center+20, center), (50, 50, 50), 1)
    cv2.line(img, (center, center-20), (center, center+20), (50, 50, 50), 1)
    
    return img

def generate_oscilloscope_lines(stereo_audio, frame_idx, hop_length, intensity, color_mode, trace_buffer, **kwargs):
    """Generate oscilloscope lines mode - connected waveform paths"""
    size = 720
    center_y = size // 2
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    if stereo_audio is None:
        return img
    
    # Extract audio chunk for this frame
    start_sample = frame_idx * hop_length
    end_sample = start_sample + hop_length
    
    if end_sample >= stereo_audio.shape[1]:
        return img
    
    # Use mix of both channels or just left if stereo
    if stereo_audio.ndim == 2:
        audio_chunk = np.mean(stereo_audio[:, start_sample:end_sample], axis=0)
    else:
        audio_chunk = stereo_audio[start_sample:end_sample]
    
    # Get oscilloscope parameters
    zoom = kwargs.get('zoom', 1.0)
    line_width = kwargs.get('line_width', 2)
    persistence = kwargs.get('persistence', 0.3)
    
    # Apply persistence
    if persistence > 0:
        img = (img * persistence).astype(np.uint8)
    
    # Scale and draw waveform
    scale = (size // 4) * intensity * zoom
    x_step = size / len(audio_chunk)
    
    prev_point = None
    
    for i, sample in enumerate(audio_chunk):
        x = int(i * x_step)
        y = int(center_y + sample * scale)
        
        # Clamp to screen bounds
        y = max(0, min(size-1, y))
        
        current_point = (x, y)
        trace_buffer.append(current_point)
        
        if prev_point is not None:
            # Color based on sample intensity and position
            if color_mode == "Oscilloscope Green":
                color = (0, int(255 * min(abs(sample) * intensity + 0.3, 1.0)), 0)
            elif color_mode == "Spectrum":
                hue = (i + frame_idx * 10) % 360
                color = hsv_to_rgb(hue, 1.0, min(abs(sample) * intensity + 0.3, 1.0))
            elif color_mode == "Heat":
                heat = min(abs(sample) * intensity + 0.3, 1.0)
                color = (int(255 * heat), int(128 * heat), 0)
            else:  # Monochrome
                val = int(255 * min(abs(sample) * intensity + 0.3, 1.0))
                color = (val, val, val)
            
            cv2.line(img, prev_point, current_point, color, line_width)
        
        prev_point = current_point
    
    # Draw grid lines
    for i in range(0, size, 60):
        cv2.line(img, (0, i), (size, i), (30, 30, 30), 1)
        cv2.line(img, (i, 0), (i, size), (30, 30, 30), 1)
    
    # Draw center line
    cv2.line(img, (0, center_y), (size, center_y), (80, 80, 80), 2)
    
    return img

# Keep all the existing generation functions (generate_circular_spectrum, etc.)
def generate_circular_spectrum(spectrum, intensity, color_mode, low_energy, mid_energy, high_energy):
    """Generate circular spectrum visualization"""
    size = 720
    center = size // 2
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create circular spectrum
    angles = np.linspace(0, 2*np.pi, len(spectrum))
    max_radius = center - 50
    
    for i, (angle, magnitude) in enumerate(zip(angles, spectrum)):
        radius = min(magnitude * intensity * 1000, max_radius)
        
        # Color based on frequency and energy
        if color_mode == "Spectrum":
            hue = (i / len(spectrum)) * 360
            color = hsv_to_rgb(hue, 1.0, min(magnitude * intensity, 1.0))
        elif color_mode == "Heat":
            heat = min(magnitude * intensity, 1.0)
            color = (int(255 * heat), int(128 * heat), 0)
        elif color_mode == "Oscilloscope Green":
            val = int(255 * min(magnitude * intensity, 1.0))
            color = (0, val, 0)
        else:  # Monochrome
            val = int(255 * min(magnitude * intensity, 1.0))
            color = (val, val, val)
        
        # Draw line from center
        end_x = int(center + radius * np.cos(angle))
        end_y = int(center + radius * np.sin(angle))
        
        cv2.line(img, (center, center), (end_x, end_y), color, 2)
    
    # Add energy-reactive center circle
    circle_radius = int(20 + (low_energy + mid_energy + high_energy) * 30)
    cv2.circle(img, (center, center), circle_radius, (255, 255, 255), -1)
    
    return img

def generate_waveform_tunnel(audio_chunk, intensity, color_mode, energy, frame_idx):
    """Generate waveform tunnel effect"""
    size = 720
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    if len(audio_chunk) == 0:
        return img
    
    # Create tunnel effect
    center = size // 2
    
    # Waveform as radius modulation
    angles = np.linspace(0, 2*np.pi, len(audio_chunk))
    base_radius = 200
    
    for i, (angle, sample) in enumerate(zip(angles, audio_chunk)):
        radius = base_radius + sample * intensity * 100 + energy * 50
        
        x = int(center + radius * np.cos(angle))
        y = int(center + radius * np.sin(angle))
        
        if 0 <= x < size and 0 <= y < size:
            # Color based on position and time
            if color_mode == "Spectrum":
                hue = (frame_idx + i) % 360
                color = hsv_to_rgb(hue, 1.0, 0.8)
            elif color_mode == "Oscilloscope Green":
                val = int(255 * min(abs(sample) * intensity + energy, 1.0))
                color = (0, val, 0)
            else:
                intensity_val = int(255 * min(abs(sample) * intensity + energy, 1.0))
                color = (intensity_val, intensity_val // 2, intensity_val)
            
            cv2.circle(img, (x, y), 3, color, -1)
    
    return img

def generate_fractal_mandala(low_energy, mid_energy, high_energy, intensity, color_mode, frame_idx):
    """Generate fractal mandala pattern"""
    size = 720
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    
    # Multiple layers of patterns
    for layer in range(3):
        layer_energy = [low_energy, mid_energy, high_energy][layer]
        base_radius = 100 + layer * 80
        
        # Number of petals based on energy
        n_petals = int(6 + layer_energy * 12)
        
        for i in range(n_petals):
            angle = (2 * np.pi * i / n_petals) + (frame_idx * 0.02)
            
            # Petal shape
            for r in range(0, int(base_radius + layer_energy * 50), 5):
                petal_angle = angle + np.sin(r * 0.1) * 0.5
                
                x = int(center + r * np.cos(petal_angle))
                y = int(center + r * np.sin(petal_angle))
                
                if 0 <= x < size and 0 <= y < size:
                    # Color based on layer and energy
                    if color_mode == "Spectrum":
                        hue = (layer * 120 + frame_idx) % 360
                        color = hsv_to_rgb(hue, 0.8, min(layer_energy * intensity, 1.0))
                    elif color_mode == "Oscilloscope Green":
                        val = int(255 * min(layer_energy * intensity, 1.0))
                        color = (0, val, 0)
                    else:
                        val = int(255 * min(layer_energy * intensity, 1.0))
                        colors = [(val, 0, 0), (0, val, 0), (0, 0, val)]
                        color = colors[layer]
                    
                    cv2.circle(img, (x, y), 2, color, -1)
    
    return img

def generate_audio_mesh(spectrum, intensity, color_mode, frame_idx):
    """Generate audio-reactive mesh"""
    size = 720
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create grid
    grid_size = 20
    for i in range(0, size, grid_size):
        for j in range(0, size, grid_size):
            # Get spectrum value for this grid position
            spec_idx = int((i + j) / (2 * grid_size) * len(spectrum)) % len(spectrum)
            height = spectrum[spec_idx] * intensity * 50
            
            # Color based on height and position
            if color_mode == "Spectrum":
                hue = (spec_idx / len(spectrum) * 360 + frame_idx) % 360
                color = hsv_to_rgb(hue, 0.8, min(height / 50, 1.0))
            elif color_mode == "Oscilloscope Green":
                val = int(255 * min(height / 50, 1.0))
                color = (0, val, 0)
            else:
                val = int(255 * min(height / 50, 1.0))
                color = (val, val, val)
            
            # Draw elevated square
            cv2.rectangle(img, (i, j), (i+grid_size, j+grid_size), color, -1)
            
            # Add 3D effect
            if height > 5:
                cv2.rectangle(img, (i+2, j+2), (i+grid_size-2, j+grid_size-2), 
                            tuple(int(c*0.7) for c in color), -1)
    
    return img

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB"""
    h = h / 360.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    
    if i % 6 == 0:
        r, g, b = v, t, p
    elif i % 6 == 1:
        r, g, b = q, v, p
    elif i % 6 == 2:
        r, g, b = p, v, t
    elif i % 6 == 3:
        r, g, b = p, q, v
    elif i % 6 == 4:
        r, g, b = t, p, v
    elif i % 6 == 5:
        r, g, b = v, p, q
    
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV

def export_video(frames, audio_data, sr, fps, resolution):
    """Export frames as video with audio"""
    # Set resolution
    if resolution == "1080p":
        width, height = 1920, 1080
    else:  # 720p
        width, height = 1280, 720
    
    # Create temporary video file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        # Resize frame to target resolution
        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)
    
    out.release()
    
    return temp_video.name

if __name__ == "__main__":
    main()
