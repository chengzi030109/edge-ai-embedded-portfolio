"""Real-world dataset loaders.

The simulator in ``tpm.signal_sim`` is fine for end-to-end demos, but a
predictive-maintenance project that only validates on its own synthetic data is
not credible. This subpackage adds loaders for public condition-monitoring
datasets so the same model code can be evaluated on data the simulator did not
generate.
"""

from __future__ import annotations

__all__ = ["cwru", "csv_replay", "phm2008"]
