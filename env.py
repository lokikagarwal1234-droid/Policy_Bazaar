"""
Environment wrappers for Atari preprocessing.
Handles frame grayscale conversion, resizing,
stacking, and reward clipping.
"""

import gymnasium as gym
import ale_py
import numpy as np
import cv2

gym.register_envs(ale_py)


class FramePreprocessor:
    """Converts a raw RGB frame to grayscale and resizes to (size, size)."""

    def __init__(self, size=84):
        self.size = size

    def process(self, frame):
        """
        Args:
            frame: np.ndarray of shape (210, 160, 3)
        Returns:
            np.ndarray of shape (size, size), dtype float32, values in [0, 1]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32)/255
        return normalized


class FrameStacker:
    """Maintains a stack of the last `n` preprocessed frames."""

    def __init__(self, n_frames=4, frame_size=84):
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.frames = None

    def reset(self, initial_frame):
        """
        Initialise the stack by repeating the first frame n times.
        Args:
            initial_frame: np.ndarray of shape (frame_size, frame_size)
        Returns:
            np.ndarray of shape (n_frames, frame_size, frame_size)
        """
        self.frames = np.stack([initial_frame]*self.n_frames, axis = 0)
        return self.frames

    def append(self, frame):
        """
        Push a new frame and drop the oldest.
        Args:
            frame: np.ndarray of shape (frame_size, frame_size)
        Returns:
            np.ndarray of shape (n_frames, frame_size, frame_size)
        """
        self.frames = np.roll(self.frames, shift = 1, axis = 0)
        self.frames[0] = frame
        return self.frames


def make_env(env_name, frame_size=84, n_frames=4):
    """
    Create the Atari environment and return it alongside
    a FramePreprocessor and FrameStacker.

    Args:
        env_name: str, Gymnasium environment ID
        frame_size: int, target spatial dimension
        n_frames: int, number of frames to stack
    Returns:
        env, preprocessor, stacker
    """
    env = gym.make(env_name, render_mode="rgb_array")
    preprocessor = FramePreprocessor(size=frame_size)
    stacker = FrameStacker(n_frames=n_frames, frame_size=frame_size)
    return env, preprocessor, stacker