
import numpy as np
from tqdm import tqdm
from buddhabrot import generate_buddhabrot
import os
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random

from scipy.signal import convolve2d

"""
# Set up parameters for resolution and number of samples
width = 2000
height = 2000
n_samples = 1000  # Number of different parameter samples
output_dir = 'buddhabrot_exploration_images'
os.makedirs(output_dir, exist_ok=True)

# Normalize array to fit colormap
def apply_colormap(buddhabrot_image, cmap_name='hot'):
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=np.min(buddhabrot_image), vmax=np.max(buddhabrot_image))
    colored_image = cmap(norm(buddhabrot_image))
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

def get_random_inside_point():
    # Sample a point in one of the two main bulbs of the Mandelbrot set
    if random.choice([True, True, True, False]):  # 75% chance of sampling from the main cardioid bulb, 25% from the main circular bulb
        # Main cardioid bulb (centered at -0.25 + 0j, radius 0.5)
        r = random.uniform(0, 0.5)
        theta = random.uniform(0, 2 * np.pi)
        inside_point = (-0.25 + 0j) + r * np.exp(1j * theta)
    else:
        # Main circular bulb (centered at -1 + 0j, radius 0.25)
        r = random.uniform(0, 0.25)
        theta = random.uniform(0, 2 * np.pi)
        inside_point = (-1 + 0j) + r * np.exp(1j * theta)
    return inside_point

def linear(x):
    return x


# Function to randomly sample parameters, including the decision for inside point
def sample_parameters():
    # Decide if inside point is None or within the main Mandelbrot bulbs
    if random.choice([True, False]):
        inside_point = None
    else:
        # Sample a point in one of the two main bulbs of the Mandelbrot set
        inside_point = get_random_inside_point()

    # Apply log scaling to the parameters

    max_steps = int(10 ** random.uniform(0, 2))
    samples = int(10 ** random.uniform(2, 4))  # Sample size range
    max_iter = int(10 ** random.uniform(1, 4))  # Max iteration range

    cmap = random.choice(['hot', 'inferno', 'plasma', 'magma', 'cividis', 'viridis', 'twilight', 'twilight_shifted',
                            'turbo', 'nipy_spectral', 'gist_ncar', 'rainbow', 'jet', 'hsv', 'ocean', 'terrain', 'gist_earth', 'cubehelix',
                            'brg', 'gnuplot', 'gnuplot2', 'CMRmap'])

    processing = random.choice([np.log1p, np.sqrt, linear])

    #lambda x : ((x - x.min()) / (x.max() - x.min())) > 0.5], lambda x : ((x - x.mean()) / x.std())) #maybe add more processing functions

    invert = random.choice([True, False])

    params = {
        'samples': samples, # Number of orbits to draw onto the image
        'max_iter': max_iter, # Number of steps before an orbit is considered to have escaped
        'max_steps': max_steps, # Number of refinement steps for the orbit, the higher, the closer the sampled point is to the border of the mandelbrot set
        #and the longer it takes to escape.
        'cmap': cmap,  # Color maps
        'inside_point': inside_point,            # Inside point for the BuddhaBrot
        'outside_point': None,                    # Outside point can be None for now
        'processing': processing,
        'invert': invert
    }
    return params

# Generate BuddhaBrot images with sampled parameters
# Generate BuddhaBrot images with sampled parameters
for i in tqdm(range(n_samples), desc="Generating BuddhaBrot samples"):
    params = sample_parameters()  # Randomly sample new parameters for each image
    samples = params['samples']
    max_iter = params['max_iter']
    max_steps = params['max_steps']
    inside_point = params['inside_point']
    cmap = params['cmap']
    processing = params['processing']
    invert = params['invert']

    with tqdm(total=samples, desc=f"Generating BuddhaBrot Image {i+1}", unit="samples") as pbar:
        # Generate BuddhaBrot with sampled parameters
        buddhabrot_image = generate_buddhabrot(width, height, samples, max_iter, max_steps, pbar, inside_point=inside_point, outside_point=None)

    # Apply processing function
    buddhabrot_image = processing(buddhabrot_image)

    # Invert the image if needed
    if invert:
        buddhabrot_image = np.max(buddhabrot_image) - buddhabrot_image

    # Rotate the image to keep visual consistency
    buddhabrot_image_flipped = np.rot90(buddhabrot_image, k=-1)

    # Apply the sampled colormap
    colored_image = apply_colormap(buddhabrot_image_flipped, cmap_name=cmap)
    img = Image.fromarray(colored_image)

    # Save the image with configuration parameters in the filename
    if inside_point is not None:
        inside_point_str = f"_ip_{inside_point.real}+{inside_point.imag}j"
    else:
        inside_point_str = "_ip_None"
    image_filename = os.path.join(output_dir, f'{i:03d}_s{samples}_mi{max_iter}_ms{max_steps}{inside_point_str}_c{cmap}_p{processing.__name__}_i{invert}.png')
    img.save(image_filename, quality=98)


print(f"{n_samples} images saved in '{output_dir}'")

"""

def apply_colormap(buddhabrot_image, cmap_name='hot'):
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=np.min(buddhabrot_image), vmax=np.max(buddhabrot_image))
    colored_image = cmap(norm(buddhabrot_image))
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

def get_random_inside_point():
    # Sample a point in one of the two main bulbs of the Mandelbrot set
    if random.choice([True, True, True, False]):  # 75% chance of sampling from the main cardioid bulb, 25% from the main circular bulb
        # Main cardioid bulb (centered at -0.25 + 0j, radius 0.5)
        r = random.uniform(0, 0.5)
        theta = random.uniform(0, 2 * np.pi)
        inside_point = (-0.25 + 0j) + r * np.exp(1j * theta)
    else:
        # Main circular bulb (centered at -1 + 0j, radius 0.25)
        r = random.uniform(0, 0.25)
        theta = random.uniform(0, 2 * np.pi)
        inside_point = (-1 + 0j) + r * np.exp(1j * theta)
    return inside_point

def get_random_render_settings():
    max_steps = int(10 ** random.uniform(0, 2))
    samples = int(10 ** random.uniform(2, 4))  # Sample size range
    max_iter = int(10 ** random.uniform(1, 4))  # Max iteration range

    cmap = random.choice(['hot', 'inferno', 'plasma', 'magma', 'cividis', 'viridis', 'twilight', 'twilight_shifted',
                            'turbo', 'nipy_spectral', 'gist_ncar', 'rainbow', 'jet', 'hsv', 'ocean', 'terrain', 'gist_earth', 'cubehelix',
                            'brg', 'gnuplot', 'gnuplot2', 'CMRmap'])

    processing = random.choice([np.log1p, np.sqrt, linear])

    invert = random.choice([True, False])

    params = {
        'samples': samples, # Number of orbits to draw onto the image
        'max_iter': max_iter, # Number of steps before an orbit is considered to have escaped
        'max_steps': max_steps, # Number of refinement steps for the orbit, the higher, the closer the sampled point is to the border of the mandelbrot set
        #and the longer it takes to escape.
        'cmap': cmap,  # Color maps
        'inside_point': get_random_inside_point(),            # Inside point for the BuddhaBrot
        'outside_point': None,                    # Outside point can be None for now
        'processing': processing,
        'invert': invert
    }
    return params


def apply_circles(img : np.array, radius : int = 1) -> np.array:
    #first generate a numpy array of size 2*radius+1
    circ = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
    original_type = img.dtype
    img = img.astype(np.float32)
    #fill the array with a circle. will this work for a circle of radius 1?
    for i in range(2*radius+1):
        for j in range(2*radius+1):
            if (i-radius)**2 + (j-radius)**2 <= radius**2:
                circ[i,j] = 1
    #then convolve the image with the circle
    ret = convolve2d(img, circ, mode='valid')
    return ret.astype(original_type)

def render(res,
           n_orbits,
           max_iter,
           orbit_refinement_steps,
           inside_point,
           outside_point,
           circle_radius,
           cmap,
           log_scale,
           invert
     ):
    # Generate BuddhaBrot with sampled parameters
    buddhabrot_image = generate_buddhabrot(res, res, n_orbits, max_iter, orbit_refinement_steps, inside_point=inside_point, outside_point=outside_point)

    return colored_image
