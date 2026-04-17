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
        """
        Fixes three issues identified in the original paper:
        1. Scale: reduced from ±10000 to ±10 — original caused value loss of 10^8
        2. Mach component: original f(v) peaked at v≈0 (rewarded low Mach / low fuel);
           now penalises deviation from v=1.0 (sonic combustor exit) with correct gradient
        3. x variable: original used x=x_norm-1 which inverted g(x); now x_norm
           used directly so solver completion (x_norm≈1) gives maximum completion reward
        """
        # Hard terminal cases
        if success and abs(v - 1.0) < 0.05:
            return 10.0
        if not success and x_norm < 0.1:
            return -10.0

        # Mach component: penalise deviation from sonic (v=1.0)
        # Mach_Mixture initialises to 10 in DARE on solver failure — clip to avoid
        # outsized penalty swamping the completion signal
        mach_err = min(abs(v - 1.0), 5.0) / 5.0  # normalised to [0, 1]
        r_mach = -5.0 * mach_err                   # range: (-5, 0]

        # Completion component: reward how far the solver ran through the engine
        r_completion = 5.0 * x_norm - 5.0          # range: (-5, 0], 0 at x_norm=1

        return r_mach + r_completion

    def _is_terminal(self, v, x_norm, success):
        target_reached = success and abs(v - 1.0) < 0.05
        unsafe = not success and x_norm < 0.1
        return target_reached or unsafe

    # ------------------------------------------------------------------
    def close(self):
        if self._eng is not None:
            self._eng.quit()
            self._eng = None
