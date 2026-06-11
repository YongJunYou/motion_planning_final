"""Load the YAML configuration (config/robot.yaml, config/task.yaml)."""
import os

import yaml

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "config")


def load_config():
    """Return (robot_cfg, task_cfg) as plain dicts."""
    with open(os.path.join(_CONFIG_DIR, "robot.yaml")) as f:
        robot = yaml.safe_load(f)
    with open(os.path.join(_CONFIG_DIR, "task.yaml")) as f:
        task = yaml.safe_load(f)
    return robot, task
