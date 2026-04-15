"""
Frame processor pipeline.

Each processor receives a BGR numpy frame and returns a (possibly annotated)
BGR frame.  Processors are chained in order; the output of one is the input
of the next.

Add object-detection / facial-recognition processors here later:

    class YoloDetector(FrameProcessor): ...
    class FaceRecognizer(FrameProcessor): ...

Then register them in MainWindow:

    self.pipeline.add(YoloDetector())
    self.pipeline.add(FaceRecognizer())
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FrameProcessor(Protocol):
    """Duck-typed interface every processor must satisfy."""

    def process(self, frame: np.ndarray) -> np.ndarray:
        ...


class PassThrough:
    """Default no-op processor – returns the frame unchanged."""

    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame


class Pipeline:
    """Runs a frame through an ordered list of processors."""

    def __init__(self) -> None:
        self._processors: list[FrameProcessor] = []

    def add(self, processor: FrameProcessor) -> None:
        self._processors.append(processor)

    def remove(self, processor: FrameProcessor) -> None:
        self._processors.remove(processor)

    def clear(self) -> None:
        self._processors.clear()

    def run(self, frame: np.ndarray) -> np.ndarray:
        for p in self._processors:
            frame = p.process(frame)
        return frame
