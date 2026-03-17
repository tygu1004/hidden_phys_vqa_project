import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.views.xform_prim_view import XformPrimView
from pxr import Gf


def randomize_color_per_instance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    color_range: tuple = (0.0, 1.0),
):
    stage = env.scene.stage
    assets = [env.scene.rigid_objects[asset_cfg.name] for asset_cfg in asset_cfgs]

    for env_id in env_ids.tolist():
        r = torch.empty(1).uniform_(color_range[0], color_range[1]).item()
        g = torch.empty(1).uniform_(color_range[0], color_range[1]).item()
        b = torch.empty(1).uniform_(color_range[0], color_range[1]).item()
        for asset in assets:
            root_path = asset.root_physx_view.prim_paths[env_id]
            color_attr_path = (
                f"{root_path}/geometry/material/Shader.inputs:diffuseColor"
            )
            color_attr = stage.GetAttributeAtPath(color_attr_path)
            if color_attr.IsValid():
                color_attr.Set(Gf.Vec3f(r, g, b))


def randomize_light_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
):
    asset: XformPrimView = env.scene[asset_cfg.name]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device
    )
    positions = env.scene.env_origins[env_ids] + rand_samples
    asset.set_world_poses(positions=positions, indices=env_ids.tolist())


def randomize_asset_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_delta_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene.rigid_objects[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    range_list = [pose_delta_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device
    )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = root_states[:, 3:7]

    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
