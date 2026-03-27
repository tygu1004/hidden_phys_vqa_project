import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.events import *
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
    asset_cfg: SceneEntityCfg,
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


def randomize_restitution_with_min_gap(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    restitution_range: tuple[float, float],
    min_gap: float,
):
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if len(asset_cfgs) != 2:
        raise ValueError("asset_cfgs must be a list of two assets.")
    if min_gap > (restitution_range[1] - restitution_range[0]):
        raise ValueError("gap can't be larger than restitution range.")

    restitution_index = 2

    asset01 = env.scene.rigid_objects[asset_cfgs[0].name]
    asset02 = env.scene.rigid_objects[asset_cfgs[1].name]

    material1 = asset01.root_physx_view.get_material_properties()
    material2 = asset02.root_physx_view.get_material_properties()

    range_width = restitution_range[1] - restitution_range[0] - min_gap
    random_samples = [
        restitution_range[0] + range_width * torch.rand(len(env_ids))
        for _ in range(len(asset_cfgs))
    ]

    random_sample_1 = torch.minimum(random_samples[0], random_samples[1])
    random_sample_2 = torch.maximum(random_samples[0], random_samples[1]) + min_gap
    swap_mask = torch.rand(len(env_ids)) > 0.5

    out1 = torch.where(swap_mask, random_sample_1, random_sample_2)
    out2 = torch.where(swap_mask, random_sample_2, random_sample_1)

    material1[env_ids, 0, restitution_index] = out1
    material2[env_ids, 0, restitution_index] = out2

    asset01.root_physx_view.set_material_properties(material1, env_ids)
    asset02.root_physx_view.set_material_properties(material2, env_ids)
