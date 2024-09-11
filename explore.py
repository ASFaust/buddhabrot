import numpy as np
from tqdm import tqdm
from bhuddabrot import generate_buddhabrot
import os
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random

# Set up parameters for resolution and number of samples
width = 2000
height = 2000
n_samples = 200  # Number of different parameter samples
output_dir = 'buddhabrot_exploration_images'
os.makedirs(output_dir, exist_ok=True)

# Normalize array to fit colormap
def apply_colormap(buddhabrot_image, cmap_name='hot'):
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=np.min(buddhabrot_image), vmax=np.max(buddhabrot_image))
    colored_image = cmap(norm(buddhabrot_image))
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

# Function to randomly sample parameters, including the decision for inside point
def sample_parameters():
    # Decide if inside point is None or within the main Mandelbrot bulbs
    if random.choice([True, False]):
        inside_point = None
    else:
        # Sample a point in one of the two main bulbs of the Mandelbrot set
        if random.choice([True, False]):
            # Main cardioid bulb (centered at -0.25 + 0j, radius 0.5)
            r = random.uniform(0, 0.5)
            theta = random.uniform(0, 2 * np.pi)
            inside_point = (-0.25 + 0j) + r * np.exp(1j * theta)
        else:
            # Main circular bulb (centered at -1 + 0j, radius 0.25)
            r = random.uniform(0, 0.25)
            theta = random.uniform(0, 2 * np.pi)
            inside_point = (-1 + 0j) + r * np.exp(1j * theta)

    params = {
        'samples': random.randint(100, 20000),  # Sample size range
        'max_iter': random.randint(500, 10000),   # Max iteration range
        'max_steps': random.randint(2, 50),       # Max steps for each point
        'cmap': random.choice(['hot', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']),  # Color maps
        'inside_point': inside_point,            # Inside point for the BuddhaBrot
        'outside_point': None                    # Outside point can be None for now
    }
    return params

# Generate BuddhaBrot images with sampled parameters
for i in tqdm(range(n_samples), desc="Generating BuddhaBrot samples"):
    params = sample_parameters()  # Randomly sample new parameters for each image
    samples = params['samples']
    max_iter = params['max_iter']
    max_steps = params['max_steps']
    inside_point = params['inside_point']
    cmap = params['cmap']

    with tqdm(total=samples, desc=f"Generating BuddhaBrot Image {i+1}", unit="samples") as pbar:
        # Generate BuddhaBrot with sampled parameters
        buddhabrot_image = generate_buddhabrot(width, height, samples, max_iter, max_steps, pbar, inside_point=inside_point, outside_point=None)
    
    # Rotate the image to keep visual consistency
    buddhabrot_image_flipped = np.rot90(buddhabrot_image, k=-1)

    # Apply the sampled colormap
    colored_image = apply_colormap(buddhabrot_image_flipped, cmap_name=cmap)
    img = Image.fromarray(colored_image)

    # Save the image
    image_filename = os.path.join(output_dir, f'buddhabrot_sample_{i+1:03d}.png')
    img.save(image_filename)

print(f"{n_samples} images saved in '{output_dir}'")

