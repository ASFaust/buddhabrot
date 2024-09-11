# Cython directives
# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
cimport cython
cimport numpy as cnp
from libc.math cimport cos, sin, M_PI
from libc.stdlib cimport malloc, free

cdef inline int mandelbrot_escape(double complex c, int max_iter, double* orbit):
    """
    Determine the escape time for the Mandelbrot set.
    Returns the number of iterations before the orbit escapes.
    Captures the entire orbit if the point escapes.
    """
    cdef double complex z = 0
    cdef int i
    for i in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return i
        orbit[2 * i] = z.real  # Store real part of the orbit
        orbit[2 * i + 1] = z.imag  # Store imaginary part of the orbit
        z = z * z + c
    return max_iter


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void binary_step(double complex c_in, double complex c_out, int max_iter, int max_steps, double* output_orbits):
    """
    Perform binary stepping to find a point close to the boundary of the Mandelbrot set.
    Start with c_in (inside) and c_out (outside), and find the boundary orbit by stepping between them.
    Capture the orbit of the point closest to the boundary.
    """
    cdef double complex c_mid
    cdef double complex c_inside = c_in
    cdef double complex c_outside = c_out
    cdef int steps = 0
    cdef int i
    
    # Pre-allocate memory for orbit (C-style malloc)
    cdef double* orbit = <double*>malloc(2 * max_iter * sizeof(double))

    # Binary stepping
    while steps < max_steps:
        c_mid = (c_inside + c_outside) / 2.0
        if mandelbrot_escape(c_mid, max_iter, orbit) == max_iter:
            c_inside = c_mid
        else:
            c_outside = c_mid
        steps += 1

    # Copy the orbit into the output memory view
    for i in range(max_iter):
        output_orbits[2 * i] = orbit[2 * i]
        output_orbits[2 * i + 1] = orbit[2 * i + 1]

    free(orbit)  # Free allocated memory

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_buddhabrot(int width, int height, int samples, int max_iter, int max_steps, pbar, 
                        inside_point=None, outside_point=None):
    """
    Generate a BuddhaBrot image by sampling orbits near the boundary of the Mandelbrot set.
    Use binary stepping to get interesting orbits and capture the full orbit.

    Optional:
        inside_point: A user-provided Python complex number that serves as the initial inside point.
                      If not provided, the function samples points from the main cardioid or the secondary bulb.
        outside_point: A user-provided Python complex number that serves as the initial outside point.
                       If not provided, the function samples points outside the Mandelbrot set.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] buddhabrot_image = np.zeros((height, width), dtype=np.float64)  # NumPy array for image

    cdef int count, orbit_len, i, j  # Declare all integer variables as C types
    cdef double x, y, theta, x_out, y_out  # Declare floating-point variables
    cdef double complex c_in, c_out  # Use Cython's C complex type
    cdef double* orbits = <double*>malloc(samples * 2 * max_iter * sizeof(double))  # C-style array
    cdef int offset 

    # Loop over samples
    for count in range(samples):
        # Determine the inside point
        if inside_point is not None:
            c_in = <double complex>inside_point  # Convert Python complex to Cython double complex
        else:
            # If no inside point provided, sample from the two main bulbs
            if np.random.uniform() < 0.75:
                # Sample from the main cardioid (75% probability)
                theta = np.random.uniform(0, 2 * M_PI)
                c_in = 0.5 * (np.exp(1j * theta) - 1)
            else:
                # Sample from the secondary circle bulb (25% probability)
                theta = np.random.uniform(0, 2 * M_PI)
                c_in = -1 + (0.25 * np.exp(1j * theta))

        # Determine the outside point
        if outside_point is not None:
            c_out = <double complex>outside_point  # Convert Python complex to Cython double complex
        else:
            # Sample angle for the outer point
            theta = np.random.uniform(0, 2 * M_PI)
            # Calculate the x_out and y_out values based on theta
            x_out = 3 * cos(theta)
            y_out = 3 * sin(theta)
            c_out = x_out + y_out * 1j

        pbar.update(1)  # Progress bar update

        offset = count * 2 * max_iter

        # Perform binary stepping to get a point near the boundary
        binary_step(c_in, c_out, max_iter, max_steps, orbits + offset)
        
        # Plot the entire orbit in the BuddhaBrot image
        for orbit_len in range(max_iter):
            x = orbits[offset + 2 * orbit_len]
            y = orbits[offset + 2 * orbit_len + 1]
            
            # Map to pixel coordinates
            i = int((x + 2.0) * width / 4.0)
            j = int((y + 2.0) * height / 4.0)
            
            if 0 <= i < width and 0 <= j < height:
                buddhabrot_image[j, i] += 1  # Increment BuddhaBrot image at the corresponding pixel
    
    # Free the allocated memory
    free(orbits)

    # Return the BuddhaBrot image
    return np.log1p(buddhabrot_image)

