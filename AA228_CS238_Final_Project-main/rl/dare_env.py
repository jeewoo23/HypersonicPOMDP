import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# matlab.engine lives in the MATLAB dist directory
_MATLAB_DIST     = r"C:\tmp\matlab_dist"
_MATLAB_EXTERN   = r"C:\Program Files\MATLAB\R2025b\extern\bin\win64"
for _p in (_MATLAB_EXTERN, _MATLAB_DIST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matlab
import matlab.engine

_DARE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "physics", "matDARE", "DARE")
)

# Equivalence ratio bounds
ER_MIN = 0.05
ER_MAX = 0.40

# Flight envelope
MACH_RANGE   = (4.0, 6.5)
ALT_RANGE_KM = (20.0, 30.0)


class DareEnv(gym.Env):
    """
    Gymnasium environment wrapping the DARE ramjet engine simulator.

    State  : altitude, equivalence_ratio, mach_mixture, x_norm, pressure, velocity, temperature
    Obs    : (pressure, velocity, temperature)  — partial observability
    Action : scalar in [-1, 1], scaled to equivalence ratio change in [-0.1, 0.1]
             new_er = (1 + action * 0.1) * current_er
    Reward : based on mach_mixture (v) and normalised exit location (x_norm mapped to x)
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=np.array([1e3,  100.0, 100.0], dtype=np.float32),
            high=np.array([1e6, 1e5,   1e5],   dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0],  dtype=np.float32),
        )

        self._eng = None
        self._er         = 0.15
        self._mach_flight = 4.0
        self._altitude    = 25.0
        self._steps       = 0
        self.max_steps    = 20

    # ------------------------------------------------------------------
    def _start_engine(self):
        if self._eng is None:
            self._eng = matlab.engine.start_matlab()
            self._eng.addpath(_DARE_DIR, nargout=0)

    def _run_dare(self, er):
        out = self._eng.dare_step(
            float(er),
            float(self._mach_flight),
            float(self._altitude),
            nargout=6,
        )
        pressure, velocity, temperature, mach_mixture, x_norm, success = out
        return (float(pressure), float(velocity), float(temperature),
                float(mach_mixture), float(x_norm), bool(success))

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._start_engine()

        rng = np.random.default_rng(seed)
        self._mach_flight = float(rng.uniform(*MACH_RANGE))
        self._altitude    = float(rng.uniform(*ALT_RANGE_KM))
        self._er          = float(rng.uniform(0.10, 0.25))
        self._steps       = 0

        pressure, velocity, temperature, _, _, _ = self._run_dare(self._er)
        obs = np.array([pressure, velocity, temperature], dtype=np.float32)
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        self._steps += 1
        delta = float(np.clip(action[0], -1.0, 1.0)) * 0.1
        self._er = float(np.clip((1.0 + delta) * self._er, ER_MIN, ER_MAX))

        pressure, velocity, temperature, mach_mixture, x_norm, success = \
            self._run_dare(self._er)

        obs     = np.array([pressure, velocity, temperature], dtype=np.float32)
        reward  = self._reward(mach_mixture, x_norm, success)
        terminated = self._is_terminal(mach_mixture, x_norm, success)
        truncated  = self._steps >= self.max_steps

        info = {
            "mach_mixture": mach_mixture,
            "x_norm": x_norm,
            "equivalence_ratio": self._er,
            "success": success,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _reward(self, v, x_norm, success):
        # x mapped so that x=0 means completed (x_norm=1), negative means short
        x = x_norm - 1.0

        if success and abs(v - 1.0) < 0.01 and abs(x) < 0.1:
            return 10000.0
        if not success and x_norm < 0.1:
            return -10000.0

        def f(v):
            if 0.99 <= v < 1.01:
                return 10000 * (0.99 - v) - 890
            elif 0.9 <= v < 0.99:
                return 2000 * (0.99 - v) - 1700
            else:
                return 11764.706 * (0.99 - v) - 10000

        def g(x):
            ax = abs(x)
            if ax < 0.1:
                return -9000 * (1 - ax) + 1000
            elif ax < 0.5:
                return -250 * (1 - ax) + 125
            else:
                return -20000 * (1 - x) + 10000

        return f(v) + g(x)

    def _is_terminal(self, v, x_norm, success):
        x = x_norm - 1.0
        target_reached = success and abs(v - 1.0) < 0.01 and abs(x) < 0.1
        unsafe = not success and x_norm < 0.1
        return target_reached or unsafe

    # ------------------------------------------------------------------
    def close(self):
        if self._eng is not None:
            self._eng.quit()
            self._eng = None
