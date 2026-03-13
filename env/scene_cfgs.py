import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_ROBOTIQ_GRIPPER_CFG


@configclass
class TableSceneCfg(InteractiveSceneCfg):
    groundplane = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    )
    sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SphereLight",
        spawn=sim_utils.SphereLightCfg(intensity=30000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 6.0, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.495)),
    )
    wall_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall0",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 6.0, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 1.0)),
    )
    wall_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall1",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 0.1, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3.0, 1.0)),
    )
    wall_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall2",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 6.0, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 0.0, 1.0)),
    )
    wall_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall3",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 0.1, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -3.0, 1.0)),
    )
    table_top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 1.2, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.2,
                restitution=0.3,
                restitution_combine_mode="max",
                friction_combine_mode="min",
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    )
    table_leg_0 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg0",
        spawn=sim_utils.CylinderCfg(
            radius=0.015,
            height=0.5,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.5, -0.26)),
    )
    table_leg_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg1",
        spawn=sim_utils.CylinderCfg(
            radius=0.015,
            height=0.5,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0.5, -0.26)),
    )
    table_leg_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg2",
        spawn=sim_utils.CylinderCfg(
            radius=0.015,
            height=0.5,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, -0.5, -0.26)),
    )
    table_leg_3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg3",
        spawn=sim_utils.CylinderCfg(
            radius=0.015,
            height=0.5,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, -0.5, -0.26)),
    )


@configclass
class FrankaRobotiqGripperTableSceneCfg(TableSceneCfg):
    robot = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    robot.init_state.pos = (0.0, 0.0, 0.0)
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -1 / 5 * np.pi,
        "panda_joint3": 0.0,
        "panda_joint4": -4 / 5 * np.pi,
        "panda_joint5": 0.0,
        "panda_joint6": 3 / 5 * np.pi,
        "panda_joint7": 0.0,
        "finger_joint": 0.0,
        "right_outer.*": 0.0,
        "left_inner.*": 0.0,
        "right_inner.*": 0.0,
    }
    robot.actuators = {
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=5200.0,
            velocity_limit_sim=2.175,
            stiffness=1100.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=720.0,
            velocity_limit_sim=2.61,
            stiffness=1000.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint"
            ],  # "right_outer_knuckle_joint" is its mimic joint
            effort_limit_sim=1650,
            velocity_limit_sim=10.0,
            stiffness=17,
            damping=0.02,
        ),
    }


@configclass
class MultiCamsFrankaRobotiqGripperTableSceneCfg(FrankaRobotiqGripperTableSceneCfg):
    cam_front_narrow = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_front_narrow",
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.0, 0.0, 0.3),
            rot=(0.5, 0.5, 0.5, 0.5),
            convention="opengl",
        ),
    )
    cam_front_wide = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_front_wide",
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=1.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.3),
            rot=(0.5, 0.5, 0.5, 0.5),
            convention="opengl",
        ),
    )
    cam_front_left_narrow = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_front_left_narrow",
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.0, -2.0, 0.3),
            rot=(0.6708, 0.6708, 0.2236, 0.2236),
            convention="opengl",
        ),
    )
    cam_front_right_narrow = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_front_right_narrow",
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.0, 2.0, 0.3),
            rot=(0.2236, 0.2236, 0.6708, 0.6708),
            convention="opengl",
        ),
    )
    cam_back_narrow = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_back_narrow",
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-2.0, 0.0, 0.3),
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="opengl",
        ),
    )
    cam_back_left_narrow = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_back_left_narrow",
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-2.0, 2.0, 0.3),
            rot=(-0.3063, -0.3063, 0.6373, 0.6373),
            convention="opengl",
        ),
    )
    cam_back_right_narrow = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_back_right_narrow",
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-2.0, -2.0, 0.3),
            rot=(0.6373, 0.6373, -0.3063, -0.3063),
            convention="opengl",
        ),
    )


@configclass
class CubeMultiCamsFrankaRobotiqGripperTableSceneCfg(
    MultiCamsFrankaRobotiqGripperTableSceneCfg
):
    cube01 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube01",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.2,
                restitution=0.5,
                friction_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.3, 0.0), roughness=0.9
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.2, 0.05)),
    )
    cube02 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube02",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.2,
                restitution=0.5,
                friction_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.3, 0.0), roughness=0.9
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.2, 0.05)),
    )


@configclass
class SphereMultiCamsFrankaRobotiqGripperTableSceneCfg(
    MultiCamsFrankaRobotiqGripperTableSceneCfg
):
    sphere01 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sphere01",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=1.0,
                restitution_combine_mode="max",
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.3, 0.0), roughness=0.9
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.2, 0.05)),
    )
    sphere02 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sphere02",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=1.0,
                restitution_combine_mode="max",
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.3, 0.0), roughness=0.9
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.2, 0.05)),
    )
