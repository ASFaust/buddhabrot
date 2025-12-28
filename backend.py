from __future__ import annotations

import numpy as np
from numba import get_thread_id, njit, prange


@njit(parallel=True)
def buddhabrot(
    samples,
    max_iter,
    escape_radius_sq,
    xmin,
    xmax,
    ymin,
    ymax,
    histograms,
    progress_counts,
    cancel_flag,
):
    n_threads, width, height = histograms.shape

    for i in prange(samples.shape[0]):
        tid = get_thread_id()
        if cancel_flag[0] != 0:
            progress_counts[tid] += 1
            continue

        hist = histograms[tid]

        c = samples[i]
        z = 0.0 + 0.0j

        for _ in range(max_iter):
            z = z * z + c
            if z.real * z.real + z.imag * z.imag > escape_radius_sq:
                break

            x = int((z.real - xmin) / (xmax - xmin) * width)
            y = int((z.imag - ymin) / (ymax - ymin) * height)

            if 0 <= x < width and 0 <= y < height:
                hist[x, y] += 1

        progress_counts[tid] += 1


# ------------------------------------------------------------
# Escape test with fixed iteration budget
# ------------------------------------------------------------


@njit
def escapes(c, max_iter, escape_radius_sq):
    z = 0.0 + 0.0j
    for _ in range(max_iter):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag > escape_radius_sq:
            return True
    return False


# ------------------------------------------------------------
# simple mandelbrot image generator for inside-outside-point sampling
# we only need to know whether a point is in the set or not
@njit(parallel=True)
def mandelbrot_image(xmin, xmax, ymin, ymax, width, height, max_iter):
    image = np.zeros((width, height), dtype=np.bool_)
    for x in prange(width):
        for y in range(height):
            cx = xmin + (x / width) * (xmax - xmin)
            cy = ymin + (y / height) * (ymax - ymin)
            c = complex(cx, cy)
            image[x, y] = not escapes(c, max_iter, 4.0)
    return image


@njit
def count_inside_outside_pairs(img):
    w, h = img.shape
    count = 0

    for x in range(1, w - 1):
        for y in range(1, h - 1):
            if img[x, y]:
                for nx in (-1, 0, 1):
                    for ny in (-1, 0, 1):
                        if nx == 0 and ny == 0:
                            continue
                        if not img[x + nx, y + ny]:
                            count += 1
    return count


@njit
def get_inside_outside_pairs(img, xmin, xmax, ymin, ymax):
    w, h = img.shape
    dx = (xmax - xmin) / w
    dy = (ymax - ymin) / h

    n_pairs = count_inside_outside_pairs(img)

    inside = np.empty(n_pairs, dtype=np.complex128)
    outside = np.empty(n_pairs, dtype=np.complex128)

    idx = 0
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            if img[x, y]:
                cx = xmin + x * dx
                cy = ymin + y * dy
                c_in = cx + 1j * cy

                for nx in (-1, 0, 1):
                    for ny in (-1, 0, 1):
                        if nx == 0 and ny == 0:
                            continue
                        if not img[x + nx, y + ny]:
                            cx_out = xmin + (x + nx) * dx
                            cy_out = ymin + (y + ny) * dy
                            inside[idx] = c_in
                            outside[idx] = cx_out + 1j * cy_out
                            idx += 1

    return inside, outside


@njit(parallel=True)
def refine_points(
    inside,
    outside,
    max_iter,
    escape_radius_sq,
    binary_steps,
    rand_pairs,
    progress_steps,
    cancel_flag,
):
    n_points = inside.shape[0]

    for step in range(binary_steps):
        for j in prange(n_points):
            c_in = inside[j]
            c_out = outside[j]

            mid = 0.5 * (c_in + c_out)

            # Chebyshev (L∞) box size
            dx = abs(c_out.real - c_in.real)
            dy = abs(c_out.imag - c_in.imag)
            s = dx if dx > dy else dy

            # sample uniformly in axis-aligned square with precomputed randomness
            rx = rand_pairs[step, j, 0]
            ry = rand_pairs[step, j, 1]
            c = mid + (rx * s) + 1j * (ry * s)

            if escapes(c, max_iter, escape_radius_sq):
                outside[j] = c
            else:
                inside[j] = c

        progress_steps[0] = step + 1
        if cancel_flag[0] != 0:
            break

    return inside, outside


def resample(inside, outside, n_samples):
    # Resample inside-outside pairs uniformly to get n_samples points
    # if we have fewer than n_samples, we append the set multiple times
    n_pairs = inside.shape[0]
    if n_pairs >= n_samples:
        indices = np.random.choice(n_pairs, n_samples, replace=False)
    else:
        n_repeats = n_samples // n_pairs
        n_extra = n_samples % n_pairs
        indices = np.concatenate(
            [np.arange(n_pairs) for _ in range(n_repeats)]
            + [np.random.choice(n_pairs, n_extra, replace=False)]
        )
    inside_resampled = inside[indices]
    outside_resampled = outside[indices]
    return inside_resampled, outside_resampled


def add_nearby_points(outside, n_per_sample, sigma):
    # outside: complex array of shape (n_samples,)
    # returns: complex array of shape (n_samples * n_per_sample,)
    # samples 2D Gaussian noise around each outside point

    n_samples = outside.shape[0]
    outside_extended = np.empty(n_samples * n_per_sample, dtype=np.complex128)

    for i in range(n_samples):
        c_out = outside[i]
        for j in range(n_per_sample):
            rx = np.random.normal(0.0, sigma)
            ry = np.random.normal(0.0, sigma)
            outside_extended[i * n_per_sample + j] = c_out + rx + 1j * ry

    return outside_extended
