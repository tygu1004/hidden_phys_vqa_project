import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

from .env_franka_drop import FrankadropSphereEnvCfg
from .env_franka_push import FrankaPushCubeEnvCfg, FrankaPushCylinderEnvCfg

gym.register(
    id="Push_Two_Cubes",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": FrankaPushCubeEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Push_Two_Cylinders",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": FrankaPushCylinderEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Drop_Two_Spheres",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": FrankadropSphereEnvCfg,
    },
    disable_env_checker=True,
)
