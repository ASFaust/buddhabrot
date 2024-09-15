# buddhabrot_ui.py

import streamlit as st
import numpy as np
from buddhabrot import generate_buddhabrot
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import time

# Function to apply colormap
def apply_colormap(buddhabrot_image, cmap_name='hot'):
    cmap = cm.get_cmap(cmap_name)
    normalized = mcolors.Normalize(vmin=np.min(buddhabrot_image), vmax=np.max(buddhabrot_image))
    colored_image = cmap(normalized(buddhabrot_image))
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

# Function to get random inside point
def get_random_inside_point():
    # Sample a point in one of the two main bulbs of the Mandelbrot set
    if np.random.choice([True, True, True, False]):  # 75% cardioid, 25% circular bulb
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 0.5)
        return complex(0.5 * (np.cos(theta) - 1.0), 0.5 * np.sin(theta))
    else:
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 0.25)
        return complex(-1.0 + 0.25 * np.cos(theta), 0.25 * np.sin(theta))

# Streamlit UI Layout
st.title("🎨 Buddhabrot Explorer")

# Sidebar Sections
st.sidebar.header("Generation Parameters")

# ### Generation Parameters Section ###
with st.sidebar:
    # Single resolution slider
    resolution = st.slider("Resolution (pixels)", min_value=500, max_value=4000, value=2000, step=100)

    # Number of samples
    samples = st.slider("Number of Samples", min_value=100, max_value=100000, value=1000, step=100)

    # Max iterations
    max_iter = st.slider("Max Iterations", min_value=50, max_value=5000, value=1000, step=50)

    # Max binary steps
    max_steps = st.slider("Max Binary Steps", min_value=10, max_value=1000, value=100, step=10)

    # Inside point selection
    inside_point_option = st.radio("Inside Point", options=["Random", "Fixed", "None"])
    if inside_point_option == "Fixed":
        fixed_real = st.text_input("Fixed Real Part", value="-0.4")
        fixed_imag = st.text_input("Fixed Imaginary Part", value="0.6")
        try:
            fixed_real = float(fixed_real)
            fixed_imag = float(fixed_imag)
            inside_point = complex(fixed_real, fixed_imag)
        except ValueError:
            st.error("Invalid fixed point coordinates. Using random instead.")
            inside_point = get_random_inside_point()
    elif inside_point_option == "None":
        inside_point = None
    else:
        inside_point = get_random_inside_point()

    # Generate button
    generate = st.button("Generate Buddhabrot")

# ### Appearance Settings Section ###
st.sidebar.header("Appearance Settings")

with st.sidebar:
    # Colormap selection
    cmap_options = [
        'hot', 'inferno', 'plasma', 'magma', 'cividis', 'viridis',
        'twilight', 'twilight_shifted', 'turbo', 'nipy_spectral',
        'gist_ncar', 'rainbow', 'jet', 'hsv', 'ocean', 'terrain',
        'gist_earth', 'cubehelix', 'brg', 'gnuplot', 'gnuplot2', 'CMRmap'
    ]
    cmap = st.selectbox("Color Map", options=cmap_options, index=0)

    # Processing function selection
    processing_options = {
        "Linear": lambda x: x,
        "Log1p": np.log1p,
        "Square Root": np.sqrt
    }
    processing_name = st.selectbox("Processing Function", options=list(processing_options.keys()))
    processing = processing_options[processing_name]

    # Inversion toggle
    invert = st.checkbox("Invert Image", value=False)

# Placeholder for the image
image_placeholder = st.empty()

if generate:
    with st.spinner("Generating Buddhabrot..."):
        start_time = time.time()
        try:
            # Initialize Streamlit progress bar
            progress_bar = st.progress(0)

            # Define a wrapper class for the progress bar
            class PBar:
                def __init__(self, total):
                    self.total = total
                    self.current = 0

                def update(self, n):
                    self.current += n
                    progress = min(self.current / self.total, 1.0)
                    progress_bar.progress(progress)

            # Create a PBar instance
            pbar = PBar(samples)

            # Generate the Buddhabrot image
            buddhabrot_image = generate_buddhabrot(
                width=resolution,
                height=resolution,
                samples=samples,
                max_iter=max_iter,
                max_steps=max_steps,
                pbar=pbar,
                inside_point=inside_point,
                outside_point=None
            )

            # Apply processing function
            buddhabrot_processed = processing(buddhabrot_image)

            # Invert the image if needed
            if invert:
                buddhabrot_processed = np.max(buddhabrot_processed) - buddhabrot_processed

            # Rotate the image for consistency
            buddhabrot_rotated = np.rot90(buddhabrot_processed, k=-1)

            # Apply colormap
            colored_image = apply_colormap(buddhabrot_rotated, cmap_name=cmap)

            # Create PIL image
            img = Image.fromarray(colored_image)

            # Display the image
            image_placeholder.image(img, caption="Buddhabrot Visualization", use_column_width=True)

            end_time = time.time()
            st.success(f"Buddhabrot generated in {end_time - start_time:.2f} seconds.")

        except MemoryError:
            st.error("MemoryError: Unable to allocate memory for orbits. Try reducing the number of samples or resolution.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
