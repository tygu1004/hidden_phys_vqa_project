import numpy as np
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass

from . import mdp
from .scene_cfgs import CubeMultiCamsFrankaRobotiqGripperTableSceneCfg


@configclass
class FrankaActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )
    finger_joint = mdp.BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class ObservationCfg:
    @configclass
    class CamsCfg(ObservationGroupCfg):
        """Camera observations."""

        cam_front_narrow = ObservationTermCfg(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_front_narrow"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        cam_front_wide = ObservationTermCfg(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_front_wide"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        cam_front_left_narrow = ObservationTermCfg(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_front_left_narrow"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        cam_front_right_narrow = ObservationTermCfg(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_front_right_narrow"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        cam_back_narrow = ObservationTermCfg(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_back_narrow"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        cam_back_left_narrow = ObservationTermCfg(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_back_left_narrow"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        cam_back_right_narrow = ObservationTermCfg(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_back_right_narrow"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    cams: CamsCfg = CamsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    cube01_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube01", body_names=".*"),
            "mass_distribution_params": (0.1, 1.0),
            "operation": "abs",
        },
    )
    cube02_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube02", body_names=".*"),
            "mass_distribution_params": (0.1, 1.0),
            "operation": "abs",
        },
    )
    cube01_pos = EventTermCfg(
        func=mdp.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube01", body_names=".*"),
            "pose_range": {"x": (0.04, 0.06), "y": (0.04, 0.06), "z": (0.0, 0.05)},
            "velocity_range": {},
        },
    )
    cube02_pos = EventTermCfg(
        func=mdp.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube02", body_names=".*"),
            "pose_range": {"x": (0.04, 0.07), "y": (-0.04, -0.06), "z": (0.0, 0.05)},
            "velocity_range": {},
        },
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class FrankaPushCubeEnvCfg(ManagerBasedRLEnvCfg):
    scene = CubeMultiCamsFrankaRobotiqGripperTableSceneCfg()
    observations = ObservationCfg()
    actions = FrankaActionCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4 * 2
        self.sim.dt = 1 / (60 * 2)

        self.num_rerenders_on_reset = 1
