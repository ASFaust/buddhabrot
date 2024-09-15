import numpy as np
from tqdm import tqdm
from buddhabrot import generate_buddhabrot
import os
from PIL import Image
import matplotlib.cm as cm  # For colormap
import matplotlib.colors as mcolors  # For normalization
import multiprocessing

# Set up parameters
width = 1000  # Increase resolution for higher quality
height = 1000
samples = 10000
max_iter = 1000
max_steps = 4
n_frames = 1000  # Number of frames for the animation

# Create directory to store frames if it doesn't exist
frame_dir = 'buddhabrot_frames_high_quality'
os.makedirs(frame_dir, exist_ok=True)

# Normalize array to fit colormap
def apply_colormap(buddhabrot_image, cmap_name='Blues'):
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=np.min(buddhabrot_image), vmax=np.max(buddhabrot_image))
    colored_image = cmap(norm(buddhabrot_image))  # Apply colormap
    return (colored_image[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit RGB image

class pbardummy:
    def update(self, x):
        pass

    def close(self):
        pass

    def set_description(self, x):
        pass

# Pre-generate BuddhaBrot frames and save them as PNGs
def generate_frame(frame):
    angle = np.linspace(0, 2 * np.pi, n_frames)[frame]  # Angle for circular motion
    inside_point = 0.03 * np.exp(1j * angle)  # Move the inside point in a circle

    # Generate the BuddhaBrot image
    buddhabrot_image = generate_buddhabrot(
        width, height, samples, max_iter, max_steps,
        pbardummy(), inside_point=inside_point, outside_point=None
    )
    buddhabrot_image_flipped = np.rot90(buddhabrot_image, k=-1)
    buddhabrot_image_flipped = np.log1p(buddhabrot_image_flipped)  # Apply log scaling to the image

    # Apply colormap and save the frame as an image using PIL
    colored_image = apply_colormap(buddhabrot_image_flipped)
    img = Image.fromarray(colored_image)

    # Save the frame
    frame_filename = os.path.join(frame_dir, f'frame_{frame:03d}.png')
    img.save(frame_filename)
    return frame_filename

if __name__ == "__main__":
    # Create a multiprocessing pool and generate frames in parallel with a progress bar
    with multiprocessing.Pool() as pool:
        # Use imap_unordered for potentially better performance
        frames = list(tqdm(
            pool.imap_unordered(generate_frame, range(n_frames)),
            total=n_frames,
            desc='Generating frames'
        ))

    # Create a video from the saved frames using ffmpeg
    output_video = 'buddhabrot_animation_high_quality.mp4'
    os.system(f"ffmpeg -r 30 -i {frame_dir}/frame_%03d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p {output_video}")

    print(f"High-quality animation saved as {output_video}")
