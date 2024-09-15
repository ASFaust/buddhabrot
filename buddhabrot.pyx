# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, fastmath=True

import numpy as np
cimport cython
cimport numpy as cnp
from libc.math cimport cos, sin, M_PI, exp, pow
from libc.stdlib cimport malloc, free, rand, RAND_MAX, srand
from libc.time cimport time

ctypedef cnp.int32_t INT_t
ctypedef cnp.float64_t DOUBLE_t

@cython.cdivision(True)
cdef inline int mandelbrot_escape(double c_real, double c_imag, int max_iter, double* orbit):
    """
    Determine the escape time for the Mandelbrot set.
    Returns the number of iterations before the orbit escapes.
    Captures the entire orbit if the point escapes.
    """
    cdef double z_real = 0.0
    cdef double z_imag = 0.0
    cdef double z_real_sq = 0.0
    cdef double z_imag_sq = 0.0
    cdef int i

    for i in range(max_iter):
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag
        if (z_real_sq + z_imag_sq) > 4.0:
            return i
        orbit[2 * i] = z_real  # Store real part of the orbit
        orbit[2 * i + 1] = z_imag  # Store imaginary part of the orbit

        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = z_real_sq - z_imag_sq + c_real

    return max_iter

@cython.cdivision(True)
@cython.inline
cdef inline void binary_step(double c_in_real, double c_in_imag, double c_out_real, double c_out_imag,
                            int max_iter, int max_steps, double* orbit_buffer):
    """
    Perform binary stepping to find a point close to the boundary of the Mandelbrot set.
    Start with c_in (inside) and c_out (outside), and find the boundary orbit by stepping between them.
    Capture the orbit of the point closest to the boundary.
    """
    cdef double c_mid_real, c_mid_imag
    cdef double c_inside_real = c_in_real
    cdef double c_inside_imag = c_in_imag
    cdef double c_outside_real = c_out_real
    cdef double c_outside_imag = c_out_imag
    cdef int steps = 0
    cdef int escape_iter

    while steps < max_steps:
        c_mid_real = 0.5 * (c_inside_real + c_outside_real)
        c_mid_imag = 0.5 * (c_inside_imag + c_outside_imag)
        escape_iter = mandelbrot_escape(c_mid_real, c_mid_imag, max_iter, orbit_buffer)
        if escape_iter == max_iter:
            c_inside_real = c_mid_real
            c_inside_imag = c_mid_imag
        else:
            c_outside_real = c_mid_real
            c_outside_imag = c_mid_imag
        steps += 1

    # After binary stepping, store the final orbit
    mandelbrot_escape(c_mid_real, c_mid_imag, max_iter, orbit_buffer)

@cython.inline
cdef inline double drand():
    """
    Generate a random double precision number between 0 and 1.
    """
    return rand() / <DOUBLE_t>RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_buddhabrot(int width, int height, int samples, int max_iter, int max_steps, pbar,
                        inside_point=None, outside_point=None):
    """
    Generate a BuddhaBrot image by sampling orbits near the boundary of the Mandelbrot set.
    Use binary stepping to get interesting orbits and capture the full orbit.
    """
    # Use int32 for the image to reduce memory usage and increase speed
    cdef cnp.ndarray[INT_t, ndim=2] buddhabrot_image = np.zeros((height, width), dtype=np.int32)
    cdef int[:, :] image_view = buddhabrot_image  # Memoryview for faster access
    cdef int count, orbit_len, i, j
    cdef double x, y, theta, x_out, y_out
    cdef double c_in_real, c_in_imag, c_out_real, c_out_imag
    cdef double* orbits = <double*>malloc(samples * 2 * max_iter * sizeof(double))
    cdef int offset

    if orbits == NULL:
        raise MemoryError("Unable to allocate memory for orbits.")

    # Initialize random seed once
    srand(<unsigned int>time(NULL))

    # Precompute constants
    cdef double scale_x = width / 4.0  # (width / 4.0)
    cdef double scale_y = height / 4.0  # (height / 4.0)

    # Precompute pi * 2
    cdef double two_pi = 2.0 * M_PI

    # Batch progress bar updates
    cdef int progress_batch = samples // 10
    if progress_batch == 0:
        progress_batch = 1
    cdef int progress_counter = 0

    cdef double real_offset = 2.0  # To map from [-2, 2] to [0, width]
    cdef double imag_offset = 2.0  # To map from [-2, 2] to [0, height]

    for count in range(samples):
        # Determine the inside point
        if inside_point is not None:
            c_in_real = (<complex>inside_point).real
            c_in_imag = (<complex>inside_point).imag
        else:
            # If no inside point provided, sample from the two main bulbs
            if drand() < 0.75:
                # Sample from the main cardioid (75% probability)
                theta = drand() * two_pi
                c_in_real = 0.5 * (cos(theta) - 1.0)
                c_in_imag = 0.5 * sin(theta)
            else:
                # Sample from the secondary circle bulb (25% probability)
                theta = drand() * two_pi
                c_in_real = -1.0 + 0.25 * cos(theta)
                c_in_imag = 0.25 * sin(theta)

        # Determine the outside point
        if outside_point is not None:
            c_out_real = (<complex>outside_point).real
            c_out_imag = (<complex>outside_point).imag
        else:
            # Sample angle for the outer point
            theta = drand() * two_pi
            # Calculate the x_out and y_out values based on theta
            x_out = 3.0 * cos(theta)
            y_out = 3.0 * sin(theta)
            c_out_real = x_out
            c_out_imag = y_out

        offset = count * 2 * max_iter

        # Perform binary stepping to get a point near the boundary
        binary_step(c_in_real, c_in_imag, c_out_real, c_out_imag, max_iter, max_steps, orbits + offset)

        # Plot the entire orbit in the BuddhaBrot image
        for orbit_len in range(max_iter):
            x = orbits[offset + 2 * orbit_len]
            y = orbits[offset + 2 * orbit_len + 1]

            # Map to pixel coordinates
            # Using integer division and precomputed scales
            i = <int>((x + real_offset) * scale_x)
            j = <int>((y + imag_offset) * scale_y)

            if (0 <= i < width) and (0 <= j < height):
                image_view[j, i] += 1

        # Update progress bar in batches to minimize Python calls
        progress_counter += 1
        if progress_counter >= progress_batch:
            pbar.update(progress_counter)
            progress_counter = 0

    # Update any remaining progress
    if progress_counter > 0:
        pbar.update(progress_counter)

    # Free the allocated memory
    free(orbits)

    # Optionally, convert the image to float if needed
    # buddhabrot_image_float = buddhabrot_image.astype(np.float64)

    # Return the BuddhaBrot image as int32
    return buddhabrot_image
