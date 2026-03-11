import argparse
import logging

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument(
    "--config_path", type=str, default="config.yaml", help="Path to the config file"
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

from configs import DataConfig, EnvConfig, RobotConfig, TaskConfig, load_config
from controllers import RobotController
from datacollector import VQADataCollector
from env import *  # noqa: F401
from planners import TaskPlanner


def main(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    task_config: TaskConfig,
    data_config: DataConfig,
    device: str,
) -> None:
    dataset_manager = VQADataCollector(
        repo_path=data_config.repo_id,
        fps=env_config.fps,
        num_envs=env_config.num_envs,
        video_keys=data_config.video_keys,
        physics_property_type=task_config.physical_property,  # "mass" or "restitution"
        vcodec=data_config.vcodec,
    )

    # Env
    env_cfg = parse_env_cfg(
        task_name=task_config.id,
        device=device,
        num_envs=env_config.num_envs,
    )
    env_cfg.scene.env_spacing = env_config.spacing
    env_cfg.episode_length_s = env_config.episode_length_s
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # obj Configs
    obj_configs = [SceneEntityCfg(name) for name in env_config.obj_names]

    # Robot
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["panda_joint.*"], body_names=["base_link"]
    )
    robot_entity_cfg.resolve(env.scene)
    robot = env.scene.articulations[robot_entity_cfg.name]

    # Robot Controller
    robot_controller = RobotController(
        env=env,
        robot_entity_cfg=robot_entity_cfg,
        ik_params=robot_config.ik_params,
        tip_offset=robot_config.tip_offset,
        device=device,
    )

    # Task Planner
    task_planner = TaskPlanner(
        env=env,
        obj_cfgs=obj_configs,
        eps=env_config.eps,
        stabilize_steps=env_config.stabilize_steps,
        hold_steps=env_config.hold_steps,
        device=device,
        phase_names=task_config.sequence,
    )

    total_collected_episodes = 0
    obs, _ = env.reset()
    trunc = torch.zeros(env_config.num_envs, dtype=torch.bool, device=device)
    while (
        simulation_app.is_running()
        and total_collected_episodes < data_config.num_episodes
    ):
        is_stabilizing, just_finished_stabilizing = (
            task_planner.update_stabilize_status()
        )

        task_planner.update_goals_if_needed(just_finished_stabilizing)

        current_goal = task_planner.get_current_targets()

        ee_tip_pos, ee_tip_quat = robot_controller.get_ee_pose()

        intermediate_pos = robot_controller.compute_intermediate_target(
            ee_tip_pos,
            current_goal[:, :3],
            robot_config.step_size,
            robot_config.noise_std,
        )

        desired_joint_pos = robot_controller.compute_ik(
            ee_tip_pos, ee_tip_quat, intermediate_pos, current_goal[:, 3:7]
        )

        ik_commands = torch.cat([intermediate_pos, current_goal[:, 3:7]], dim=1)

        # For stabilizing objs, keep current position.
        desired_joint_pos[is_stabilizing] = robot.data.joint_pos[is_stabilizing, :7]
        desired_joint_pos = torch.cat([desired_joint_pos, current_goal[:, [7]]], dim=1)

        desired_ee_action = torch.cat([ik_commands, current_goal[:, [7]]], dim=1)
        desired_ee_action[is_stabilizing, :3] = ee_tip_pos[is_stabilizing]
        desired_ee_action[is_stabilizing, 3:7] = ee_tip_quat[is_stabilizing]

        # --- Data Collection ---
        data_collecting_envs_idx = (~is_stabilizing).nonzero().flatten()
        dataset_manager.add_frame(observation=obs, env_indices=data_collecting_envs_idx)

        obs, _, _, trunc, _ = env.step(desired_joint_pos)

        # --- Progress Checking ---
        task_planner.check_reach_and_advance(ee_tip_pos, ik_commands[:, 0:3])

        # --- Reset/Truncation ---
        if trunc.any():
            trunc_env_idx = trunc.nonzero().flatten().tolist()
            if trunc_env_idx:
                dataset_manager.save_episode(
                    env=env, env_indices=trunc_env_idx, obj_cfgs=obj_configs
                )
                total_collected_episodes += len(trunc_env_idx)
                logging.info(
                    f"[Success] Saved {len(trunc_env_idx)} episodes. Total: {total_collected_episodes} / {data_config.num_episodes}"
                )

            logging.info(
                f"done. {total_collected_episodes} / {data_config.num_episodes}"
            )
            task_planner.reset_envs(trunc)

    env.close()


if __name__ == "__main__":
    robot_config, env_config, task_config, data_config = load_config(
        path=args_cli.config_path
    )

    main(
        robot_config=robot_config,
        env_config=env_config,
        task_config=task_config,
        data_config=data_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    simulation_app.close()
