import os
from dataclasses import dataclass

import yaml


@dataclass
class RobotConfig:
    type: str
    tip_offset: list[float]
    ik_params: dict[str, float]
    step_size: float
    noise_std: float


@dataclass
class EnvConfig:
    num_envs: int
    fps: int
    spacing: float
    obj_names: list[str]
    stabilize_steps: int
    hold_steps: int
    eps: float
    episode_length_s: int


@dataclass
class TaskConfig:
    id: str
    sequence: list[str]
    physical_property: str


@dataclass
class DataConfig:
    repo_id: str
    num_episodes: int
    video_keys: list[str]
    vcodec: str


def load_config(path: str = "config.yaml"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, path)
    with open(full_path, "r") as f:
        config = yaml.safe_load(f)

    # Post-process features shapes (list -> tuple)
    if "data" in config and "features" in config["data"]:
        for value in config["data"]["features"].values():
            if "shape" in value and isinstance(value["shape"], list):
                value["shape"] = tuple(value["shape"])

    robot_config = RobotConfig(**config["robot"])
    env_config = EnvConfig(**config["env"])
    task_config = TaskConfig(**config["task"])
    data_config = DataConfig(**config["data"])

    return robot_config, env_config, task_config, data_config
