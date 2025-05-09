import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import io
import pandas as pd

# Set page config
st.set_page_config(page_title="DJ Visual Methods Comparison", layout="wide")
st.title("DJ Show Visualization Methods Comparison")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
    
    # Visualization selection
    st.subheader("Select visualizations")
    show_pyqtgraph = st.checkbox("PyQtGraph with OpenGL", value=True)
    show_moderngl = st.checkbox("ModernGL Spectrogram", value=True)
    show_circle = st.checkbox("Circular Visualizer", value=True)
    show_p5 = st.checkbox("p5 Visualizer", value=True)
    
    # Parameters
    fft_size = st.slider("FFT Size", 512, 4096, 2048, 512)
    color_theme = st.selectbox("Color Theme", ["viridis", "plasma", "inferno", "magma"])

# Load audio & create visualizations
def process_audio(file, fft_size=2048):
    # Load audio data
    y, sr = librosa.load(file, sr=None)
    return y, sr

# Create visualizations based on the audio data
def create_visualizations(y, sr, fft_size):
    visuals = {}
    
    # PyQtGraph OpenGL style - 3D mesh visualization
    if show_pyqtgraph:
        # Create 3D surface with plotly (simulating PyQtGraph)
        D = np.abs(librosa.stft(y, n_fft=fft_size))
        D_reduced = D[::4, ::8]  # Reduce dimensionality
        
        x, y_mesh = np.meshgrid(
            np.arange(D_reduced.shape[1]), 
            np.arange(D_reduced.shape[0])
        )
        
        fig = go.Figure(data=[go.Surface(
            z=D_reduced, x=x, y=y_mesh, colorscale=color_theme)])
        
        fig.update_layout(
            title="3D Audio Surface (PyQtGraph Style)",
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Frequency",
                zaxis_title="Magnitude"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        visuals["pyqtgraph"] = fig
    
    # ModernGL style - Spectrogram
    if show_moderngl:
        fig, ax = plt.subplots(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=fft_size)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax, cmap=color_theme)
        ax.set_title('Spectrogram (ModernGL Style)')
        plt.colorbar(img, ax=ax, format="%+2.0f dB")
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        visuals["moderngl"] = buf
    
    # Circular visualizer
    if show_circle:
        width, height = 600, 600
        image = Image.new('RGBA', (width, height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(image)
        
        # Extract audio features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Draw concentric circles based on audio features
        center_x, center_y = width//2, height//2
        cmap = plt.get_cmap(color_theme)
        
        for i, strength in enumerate(onset_env[:15]):
            radius = int(50 + strength * 250)
            color_val = i/15
            color = tuple(int(c*255) for c in cmap(color_val)[:3]) + (200,)
            
            draw.ellipse(
                (center_x-radius, center_y-radius, 
                 center_x+radius, center_y+radius),
                outline=color, width=3
            )
        
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        visuals["circle"] = buf
    
    # p5 style frequency bars
    if show_p5:
        # Calculate frequency data
        D = np.abs(librosa.stft(y, n_fft=fft_size))
        freq_data = np.mean(D, axis=1)
        freq_data = freq_data / np.max(freq_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=np.arange(len(freq_data[:64])),
            y=freq_data[:64],
            marker_color=np.arange(64),
            marker_colorscale=color_theme
        ))
        
        fig.update_layout(
            title="Frequency Bars (p5 Style)",
            xaxis_title="Frequency Band",
            yaxis_title="Magnitude",
            template="plotly_dark"
        )
        
        visuals["p5"] = fig
        
    return visuals

# Main application flow
if uploaded_file:
    st.audio(uploaded_file)
    
    try:
        with st.spinner("Processing audio..."):
            y, sr = process_audio(uploaded_file)
            visuals = create_visualizations(y, sr, fft_size)
        
        # Display method comparison table
        st.subheader("Visualization Methods Comparison")
        comparison = pd.DataFrame({
            'Method': ['PyQtGraph OpenGL', 'ModernGL', 'AudioVisualCircle', 'p5', 'Realtime-Audio-Visualizer'],
            'Best For': ['3D immersive visuals', 'High-performance spectrograms', 'Circular motion graphics', 'Creative coding', 'Detailed analysis'],
            'Performance': ['High', 'Very High', 'Medium', 'Medium', 'High'],
            'Visualization Type': ['3D Surface', 'Spectrogram', 'Circular', 'Custom Graphics', 'Waveform/FFT']
        })
        st.dataframe(comparison)
        
        # Display visualizations
        if visuals:
            st.header("Visualization Examples")
            
            # Create columns for the visualizations
            cols = st.columns(2)
            col_idx = 0
            
            # Display each visualization
            if "pyqtgraph" in visuals:
                with cols[col_idx % 2]:
                    st.subheader("PyQtGraph OpenGL Style")
                    st.plotly_chart(visuals["pyqtgraph"], use_container_width=True)
                col_idx += 1
                
            if "moderngl" in visuals:
                with cols[col_idx % 2]:
                    st.subheader("ModernGL Spectrogram")
                    st.image(visuals["moderngl"])
                col_idx += 1
                
            if "circle" in visuals:
                with cols[col_idx % 2]:
                    st.subheader("Circular Visualizer")
                    st.image(visuals["circle"])
                col_idx += 1
                
            if "p5" in visuals:
                with cols[col_idx % 2]:
                    st.subheader("p5 Style Visualization")
                    st.plotly_chart(visuals["p5"], use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    # Display welcome information
    st.markdown("""
    ## Compare DJ Show Visualization Methods
    
    Upload an audio file to see how it would look using different Python visualization packages:
    
    1. **PyQtGraph with OpenGL** - Creates immersive 3D visualizations with excellent performance[1][11]
    2. **ModernGL** - Provides GPU-accelerated graphics for high-performance spectrograms[2][12][16]
    3. **AudioVisualCircle** - Creates circular visualizations that pulse with the music[3][4]
    4. **p5** - Offers creative freedom for unique visual styles based on Processing[6][14]
    5. **Python-Realtime-Audio-Visualizer** - Multi-process design for smooth performance[7]
    
    The sidebar controls let you select which visualizations to display and customize parameters.
    """)
