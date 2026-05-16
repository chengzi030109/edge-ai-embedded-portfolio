"""Tests for the synthetic motor vibration source."""

import numpy as np

from tpm.signal_sim import MotorSignalSimulator, SignalConfig, state_schedule


def test_fault_states_change_signal_energy():
    """Fault states should be measurably different from normal operation."""

    simulator = MotorSignalSimulator(SignalConfig(seed=123))
    normal = simulator.read(256, "normal")
    imbalance = simulator.read(256, "imbalance")
    rubbing = simulator.read(256, "rubbing")
    bearing = simulator.read(256, "bearing")

    normal_rms = float(np.sqrt(np.mean(normal * normal)))
    assert float(np.sqrt(np.mean(imbalance * imbalance))) > normal_rms
    assert float(np.max(rubbing)) > float(np.max(normal))
    assert float(np.std(bearing)) > normal_rms * 0.9


def test_state_schedule_contains_faults():
    """The demo schedule should move from normal into fault states."""

    states = list(state_schedule(duration_s=4.0, window_s=0.16))

    assert states[0] == "normal"
    assert "imbalance" in states
    assert "rubbing" in states
    assert "bearing" in states
