import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


class RobotController:
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        robot_entity_cfg: SceneEntityCfg,
        ik_params: dict,
        tip_offset: list[float],
        device: str,
    ):
        self.env = env
        self.robot_entity_cfg = robot_entity_cfg
        self.device = device
        self.num_envs = env.num_envs

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params=ik_params,
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.num_envs, device=self.device
        )

        self.robot = env.scene.articulations[robot_entity_cfg.name]

        if self.robot.is_fixed_base:
            self.ee_link_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_link_idx = robot_entity_cfg.body_ids[0]

        self.tip_offset = torch.tensor(tip_offset, device=device).repeat(
            self.num_envs, 1
        )
        self.env_origins = env.scene.env_origins

    def get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_pose_w = self.robot.data.body_link_pose_w[
            :, self.robot_entity_cfg.body_ids[0]
        ]
        base_pos = base_pose_w[:, :3] - self.env_origins
        base_quat = base_pose_w[:, 3:7]
        rotated_offset = quat_apply(base_quat, self.tip_offset)
        return base_pos + rotated_offset, base_quat

    def compute_intermediate_target(
        self,
        current_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        step_size: float,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        direction = goal_pos - current_pos
        dist = torch.norm(direction, dim=1, keepdim=True)
        direction = direction / (dist + 1e-6)
        intermediate_pos = current_pos + direction * torch.clamp(dist, max=step_size)

        if noise_std > 0:
            pos_noise = torch.randn_like(intermediate_pos) * noise_std
            intermediate_pos = intermediate_pos + pos_noise

        return intermediate_pos

    def compute_ik(self, ee_pos, ee_quat, target_pos, target_quat) -> torch.Tensor:
        ik_commands = torch.cat([target_pos, target_quat], dim=1)
        self.diff_ik_controller.set_command(ik_commands)
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_link_idx, :, self.robot_entity_cfg.joint_ids
        ]
        return self.diff_ik_controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
