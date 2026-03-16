import random

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.views.xform_prim_view import XformPrimView
from pxr import Gf, UsdGeom


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
    position_range: dict[str, tuple[float, float]],
):
    stage = env.scene.stage
    asset: XformPrimView = env.scene[asset_cfg.name]

    range_x = position_range.get("x")
    range_y = position_range.get("y")
    range_z = position_range.get("z")

    for env_id in env_ids.tolist():
        prim = stage.GetPrimAtPath(asset.prim_paths[env_id])
        xform = UsdGeom.Xformable(prim)

        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if not translate_op:
            translate_op = xform.AddTranslateOp()

        rand_x = random.uniform(range_x[0], range_x[1])
        rand_y = random.uniform(range_y[0], range_y[1])
        rand_z = random.uniform(range_z[0], range_z[1])

        translate_op.Set(Gf.Vec3d(rand_x, rand_y, rand_z))
