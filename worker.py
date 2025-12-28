from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import get_num_threads
from PySide6 import QtCore

from backend import (
    buddhabrot,
    add_nearby_points,
    get_inside_outside_pairs,
    mandelbrot_image,
    refine_points,
    resample,
)


@dataclass(frozen=True)
class GenerationParams:
    samples: int
    max_iter: int
    resolution: int
    mandelbrot_resolution: int
    refine_steps: int
    nearby_points_per_boundary: int
    nearby_distance_value: float
    seed: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float


STAGE_IDLE = 0
STAGE_MANDELBROT = 1
STAGE_SAMPLE_PAIRS = 2
STAGE_REFINE = 3
STAGE_BUDDHABROT = 4
STAGE_COMBINE = 5


class BuddhabrotWorker(QtCore.QObject):
    finished = QtCore.Signal(np.ndarray)
    progress = QtCore.Signal(int)
    error = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(
        self,
        params: GenerationParams,
        progress_stage: np.ndarray,
        progress_steps: np.ndarray,
        progress_samples: np.ndarray,
        cancel_flag: np.ndarray,
    ):
        super().__init__()
        self._params = params
        self._progress_stage = progress_stage
        self._progress_steps = progress_steps
        self._progress_samples = progress_samples
        self._cancel_flag = cancel_flag

    @QtCore.Slot()
    def run(self) -> None:
        try:
            params = self._params
            width = params.resolution
            height = params.resolution
            np.random.seed(int(params.seed))

            self._progress_stage[0] = STAGE_MANDELBROT
            self._progress_steps[0] = 0
            self._progress_samples[:] = 0
            self._cancel_flag[0] = 0
            self.progress.emit(5)
            sample_grid = min(params.mandelbrot_resolution, width, height)
            mandelbrot_img = mandelbrot_image(
                params.xmin,
                params.xmax,
                params.ymin,
                params.ymax,
                sample_grid,
                sample_grid,
                params.max_iter,
            )

            if QtCore.QThread.currentThread().isInterruptionRequested():
                self.canceled.emit()
                return

            self._progress_stage[0] = STAGE_SAMPLE_PAIRS
            self.progress.emit(25)
            inside, outside = get_inside_outside_pairs(
                mandelbrot_img, params.xmin, params.xmax, params.ymin, params.ymax
            )
            inside, outside = resample(inside, outside, params.samples)

            if QtCore.QThread.currentThread().isInterruptionRequested():
                self.canceled.emit()
                return

            self._progress_stage[0] = STAGE_REFINE
            self._progress_steps[0] = 0
            self.progress.emit(45)
            rand_pairs = (
                np.random.random((params.refine_steps, inside.shape[0], 2)) - 0.5
            )
            inside, outside = refine_points(
                inside,
                outside,
                params.max_iter,
                4.0,
                binary_steps=params.refine_steps,
                rand_pairs=rand_pairs,
                progress_steps=self._progress_steps,
                cancel_flag=self._cancel_flag,
            )

            if QtCore.QThread.currentThread().isInterruptionRequested():
                self._cancel_flag[0] = 1
                self.canceled.emit()
                return

            self._progress_stage[0] = STAGE_BUDDHABROT
            self._progress_samples[:] = 0
            self.progress.emit(65)
            if params.nearby_points_per_boundary > 0:
                distance_factor = 1.0 / max(params.nearby_distance_value, 1e-9)
                nearby = add_nearby_points(
                    outside, params.nearby_points_per_boundary, distance_factor
                )
                outside = np.concatenate([outside, nearby])
            n_threads = get_num_threads()
            histograms = np.zeros((n_threads, width, height), dtype=np.int32)
            buddhabrot(
                outside,
                params.max_iter,
                4.0,
                params.xmin,
                params.xmax,
                params.ymin,
                params.ymax,
                histograms,
                self._progress_samples,
                self._cancel_flag,
            )

            if QtCore.QThread.currentThread().isInterruptionRequested():
                self._cancel_flag[0] = 1
                self.canceled.emit()
                return

            self._progress_stage[0] = STAGE_COMBINE
            self.progress.emit(90)
            final_histogram = np.zeros((width, height), dtype=histograms.dtype)
            for t in range(n_threads):
                final_histogram += histograms[t]

            self.progress.emit(100)
            self._progress_stage[0] = STAGE_IDLE
            self.finished.emit(final_histogram)
        except Exception as exc:
            self.error.emit(str(exc))
