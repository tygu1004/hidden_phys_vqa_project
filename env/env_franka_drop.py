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
from .scene_cfgs import SphereMultiCamsFrankaRobotiqGripperTableSceneCfg


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
    light_pos = EventTermCfg(
        func=mdp.randomize_light_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sphere_light"),
            "pose_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (2.0, 2.5)},
        },
    )
    spheres_restitution = EventTermCfg(
        func=mdp.randomize_restitution_with_min_gap,
        mode="reset",
        params={
            "asset_cfgs": [
                SceneEntityCfg("sphere01", body_names=".*"),
                SceneEntityCfg("sphere02", body_names=".*"),
            ],
            "restitution_range": (0.0, 1.0),
            "min_gap": 0.2,
        },
    )
    sphere01_pos = EventTermCfg(
        func=mdp.randomize_asset_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sphere01", body_names=".*"),
            "pose_delta_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (0.0, 0.0),
            },
        },
    )
    sphere02_pos = EventTermCfg(
        func=mdp.randomize_asset_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sphere02", body_names=".*"),
            "pose_delta_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (0.0, 0.0),
            },
        },
    )
    spheres_color = EventTermCfg(
        func=mdp.randomize_color_per_instance,
        mode="reset",
        params={
            "asset_cfgs": [
                SceneEntityCfg("sphere01", body_names=".*"),
                SceneEntityCfg("sphere02", body_names=".*"),
            ],
            "color_range": (0.0, 1.0),
        },
    )
    wall_0_color = EventTermCfg(
        func=mdp.randomize_color_per_instance,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("wall_0", body_names=".*")],
            "color_range": (0.0, 1.0),
        },
    )
    wall_1_color = EventTermCfg(
        func=mdp.randomize_color_per_instance,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("wall_1", body_names=".*")],
            "color_range": (0.0, 1.0),
        },
    )
    wall_2_color = EventTermCfg(
        func=mdp.randomize_color_per_instance,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("wall_2", body_names=".*")],
            "color_range": (0.0, 1.0),
        },
    )
    wall_3_color = EventTermCfg(
        func=mdp.randomize_color_per_instance,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("wall_3", body_names=".*")],
            "color_range": (0.0, 1.0),
        },
    )
    table_top_color = EventTermCfg(
        func=mdp.randomize_color_per_instance,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("table_top", body_names=".*")],
            "color_range": (0.0, 1.0),
        },
    )
    floor_color = EventTermCfg(
        func=mdp.randomize_color_per_instance,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("floor", body_names=".*")],
            "color_range": (0.0, 1.0),
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
class FrankadropSphereEnvCfg(ManagerBasedRLEnvCfg):
    scene = SphereMultiCamsFrankaRobotiqGripperTableSceneCfg()
    scene.replicate_physics = False
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
