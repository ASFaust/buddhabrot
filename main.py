import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from buddhabrot import generate_buddhabrot

width = 1000
height = 1000
samples = 10000
max_iter = 1000
max_steps = 4

# Generate BuddhaBrot image
with tqdm(total=samples, desc="Generating BuddhaBrot", unit="samples") as pbar:
    buddhabrot_image = generate_buddhabrot(width, height, samples, max_iter, max_steps, pbar, inside_point = 0.01+0.01j, outside_point = None)

# Flip the image 90 degrees clockwise
buddhabrot_image_flipped = np.rot90(buddhabrot_image, k=-1)

# Display the image with equal axis scaling
plt.imshow(buddhabrot_image_flipped, cmap='hot')
plt.colorbar()
plt.axis('equal')  # Ensure equal scaling for the x and y axes
plt.show()



