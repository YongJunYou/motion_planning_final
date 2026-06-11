"""M6: virtual force sensor readout (wrist joint force/torque). Step 2.

Reads the wrist joint force/torque (articulation joint force / effort sensor),
which measures the load through the wrist regardless of pad compliance. The
exact sensor API varies by Isaac Sim version, so confirm against the installed
version. See ROADMAP.md, Track B.
"""


def read_wrist_wrench(*args, **kwargs):
    raise NotImplementedError("Virtual force sensor is implemented in Step 2 (M6).")
