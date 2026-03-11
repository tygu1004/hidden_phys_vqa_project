import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def get_pushing_ee_goals(
    env: ManagerBasedRLEnv,
    obj_cfgs: list[SceneEntityCfg],
    ee_quat=[0.0, 1.0, 0.0, 0.0],
) -> torch.Tensor:
    """
    output shape : (num_ee_goal_pos, num_env, 8(pos, quat, gripper))
    """
    env_origins = env.scene.env_origins
    ee_goals = []
    ee_quat = torch.tensor(ee_quat, device=env.device).repeat(env.num_envs, 1)
    for obj_cfg in obj_cfgs:
        obj = env.scene.rigid_objects[obj_cfg.name]
        current_obj_pos = obj.data.root_link_pos_w - env_origins
        approach_pos = current_obj_pos.clone()
        approach_pos[:, [0]] -= 0.1
        target_pos = current_obj_pos.clone()
        # target_pos[:, [0]] += 0.1
        gripper_pos = torch.ones(env.num_envs, 1, device=env.device)

        ee_goals.append(torch.cat([approach_pos, ee_quat, gripper_pos], dim=1))
        ee_goals.append(torch.cat([target_pos, ee_quat, gripper_pos], dim=1))
        ee_goals.append(torch.cat([approach_pos, ee_quat, gripper_pos], dim=1))

    return torch.stack(ee_goals)


def get_stacking_ee_goals(
    env: ManagerBasedRLEnv,
    upper_obj_cfg: SceneEntityCfg,
    under_obj_cfg: SceneEntityCfg,
    ee_quat=[0.0, 1.0, 0.0, 0.0],
    height=0.10,
) -> torch.Tensor:
    """
    output shape : (num_ee_goal_pos, num_env, 8(pos, quat, gripper))
    """
    env_origins = env.scene.env_origins

    ee_quat = torch.tensor(ee_quat, device=env.device).repeat(env.num_envs, 1)
    upper_obj = env.scene.rigid_objects[upper_obj_cfg.name]
    under_obj = env.scene.rigid_objects[under_obj_cfg.name]

    upper_obj_root_link_pos = upper_obj.data.root_link_pos_w - env_origins
    under_obj_root_link_pos = under_obj.data.root_link_pos_w - env_origins

    align_pose_point = upper_obj_root_link_pos.clone()
    align_pose_point[:, [2]] += height

    grasp_point = upper_obj_root_link_pos.clone()
    grasp_point[:, [2]] -= 0.02

    release_point_1 = under_obj_root_link_pos.clone()
    release_point_1[:, [2]] += height

    release_point_2 = under_obj_root_link_pos.clone()
    release_point_2[:, [2]] += height - 0.04

    move_away_point_1 = release_point_2.clone()
    move_away_point_1[:, [2]] += 0.1

    move_away_point_2 = move_away_point_1.clone()
    move_away_point_2[:, [0]] -= 0.1

    ee_goals = [
        # aligning ee and opening gripper
        torch.cat(
            [
                align_pose_point,
                ee_quat,
                torch.zeros(env.num_envs, 1, device=env.device),
            ],
            dim=1,
        ),
        # grasping the upper cube
        torch.cat(
            [grasp_point, ee_quat, torch.zeros(env.num_envs, 1, device=env.device)],
            dim=1,
        ),
        torch.cat(
            [grasp_point, ee_quat, torch.ones(env.num_envs, 1, device=env.device)],
            dim=1,
        ),
        # releaing the upper cube on the under cube
        torch.cat(
            [release_point_1, ee_quat, torch.ones(env.num_envs, 1, device=env.device)],
            dim=1,
        ),
        torch.cat(
            [release_point_2, ee_quat, torch.ones(env.num_envs, 1, device=env.device)],
            dim=1,
        ),
        torch.cat(
            [release_point_2, ee_quat, torch.zeros(env.num_envs, 1, device=env.device)],
            dim=1,
        ),
        # moving away from a release point
        torch.cat(
            [
                move_away_point_1,
                ee_quat,
                torch.zeros(env.num_envs, 1, device=env.device),
            ],
            dim=1,
        ),
        torch.cat(
            [
                move_away_point_2,
                ee_quat,
                torch.zeros(env.num_envs, 1, device=env.device),
            ],
            dim=1,
        ),
    ]

    return torch.stack(ee_goals)


def is_stacked(
    env: ManagerBasedRLEnv,
    obj_cfgs: list[SceneEntityCfg],
    obj_size: float = 0.05,
    xy_tol: float = 0.02,
    z_tol: float = 0.015,
) -> torch.Tensor:
    """
    순서에 상관없이 두 object 중 하나가 다른 하나 위에 쌓여있는지 확인하는 종료 조건.
    """
    env_origins = env.scene.env_origins

    if len(obj_cfgs) != 2:
        raise ValueError(
            "obj_cfgs should contain exactly 2 objects for stacking check."
        )

    obj1: RigidObject = env.scene[obj_cfgs[0].name]
    obj2: RigidObject = env.scene[obj_cfgs[1].name]

    pos1 = obj1.data.root_link_pos_w - env_origins
    pos2 = obj2.data.root_link_pos_w - env_origins

    xy_distance = torch.norm(pos1[:, :2] - pos2[:, :2], dim=-1)
    z_distance = torch.abs(pos1[:, 2] - pos2[:, 2])
    is_xy_aligned = xy_distance < xy_tol
    is_z_stacked = torch.abs(z_distance - obj_size) < z_tol

    return is_xy_aligned & is_z_stacked


def get_dropping_ee_goals(
    env: ManagerBasedRLEnv,
    obj_cfgs: list[SceneEntityCfg],
    ee_quat=[0.0, 1.0, 0.0, 0.0],
    height=0.1,
):
    """
    dropping task를 위한 EE 목표 위치 생성 함수. obj_cfgs는 드롭할 객체들의 cfg 리스트입니다.
    output shape : (num_ee_goal_pos, num_env, 8(pos, quat, gripper))
    """
    env_origins = env.scene.env_origins
    ee_goals = []
    ee_quat = torch.tensor(ee_quat, device=env.device).repeat(env.num_envs, 1)
    for obj_cfg in obj_cfgs:
        obj = env.scene.rigid_objects[obj_cfg.name]
        current_obj_pos = obj.data.root_link_pos_w - env_origins

        align_point = current_obj_pos.clone()
        align_point[:, [2]] += 0.05

        grasp_point = current_obj_pos.clone()
        # grasp_point[:, [2]] -= 0.02

        drop_pos = current_obj_pos.clone()
        drop_pos[:, [2]] += height

        # aligning ee and opening gripper
        ee_goals.append(
            torch.cat(
                [align_point, ee_quat, torch.zeros(env.num_envs, 1, device=env.device)],
                dim=1,
            )
        )
        # grasping the upper cube
        ee_goals.append(
            torch.cat(
                [grasp_point, ee_quat, torch.zeros(env.num_envs, 1, device=env.device)],
                dim=1,
            )
        )
        ee_goals.append(
            torch.cat(
                [grasp_point, ee_quat, torch.ones(env.num_envs, 1, device=env.device)],
                dim=1,
            )
        )
        # moving to the drop position and releasing the object
        ee_goals.append(
            torch.cat(
                [drop_pos, ee_quat, torch.ones(env.num_envs, 1, device=env.device)],
                dim=1,
            )
        )
        ee_goals.append(
            torch.cat(
                [drop_pos, ee_quat, torch.zeros(env.num_envs, 1, device=env.device)],
                dim=1,
            )
        )

    return torch.stack(ee_goals)
