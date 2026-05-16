"""TinyML predictive maintenance prototype.

Package responsibilities:
- ``signal_sim`` generates hardware-free vibration windows
- ``features`` turns raw windows into compact model inputs
- ``model`` implements the tiny anomaly detector
- ``rtos_sim`` connects the stages into an RTOS-style node loop
- ``telemetry`` writes gateway-readable messages
- ``evaluation`` computes reportable anomaly-detection metrics
- ``config`` loads reproducible project settings
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
