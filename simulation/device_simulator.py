"""
simulation/device_simulator.py — STAGES 20, 23

Purpose
-------
Since no physical IoT devices are available, this module:
  1. Maps processed test-set rows to simulated devices (config.SIMULATED_DEVICES).
  2. Generates one "event" at a time by drawing a real row from the
     processed test split (never fabricated feature values) and assigning
     it to a device.

All output is clearly labeled SIMULATION MODE and must never be presented
as real physical network traffic (Section 23).

Required packages
------------------
pip install pandas numpy

How to run
----------
python -m simulation.device_simulator   (prints a few sample events)

Expected output
---------------
A stream of dict events: {"device": ..., "features": {...}, "true_label": ...}
"""

import os
import random

import pandas as pd

import config


class DeviceEventSimulator:
    MODE_LABEL = "SIMULATION MODE"

    def __init__(self, test_csv: str = None, seed: int = config.RANDOM_STATE):
        self.test_csv = test_csv or os.path.join(config.PROCESSED_DATA_DIR, "test.csv")
        self.rng = random.Random(seed)
        self._df = None

    def _ensure_loaded(self):
        if self._df is None:
            if not os.path.exists(self.test_csv):
                raise FileNotFoundError(
                    f"{self.test_csv} not found. Run preprocessing/preprocess.py first "
                    f"to produce processed test data for simulation."
                )
            self._df = pd.read_csv(self.test_csv)

    def next_event(self) -> dict:
        """Draws one real row from the processed test set and assigns it to
        a randomly chosen simulated device. Feature values are REAL (from
        the dataset), only the device assignment is simulated."""
        self._ensure_loaded()
        row = self._df.sample(n=1, random_state=self.rng.randint(0, 10_000)).iloc[0]
        device = self.rng.choice(config.SIMULATED_DEVICES)
        row_dict = row.to_dict()
        return {
            "mode": self.MODE_LABEL,
            "device": device,
            "row": row_dict,
        }

    def device_status(self) -> dict:
        """Simple simulated device status board (all devices 'online' by
        default in this software-only simulation)."""
        return {d: "online (simulated)" for d in config.SIMULATED_DEVICES}


if __name__ == "__main__":
    sim = DeviceEventSimulator()
    try:
        for _ in range(3):
            print(sim.next_event())
    except FileNotFoundError as e:
        print(e)
