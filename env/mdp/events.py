import omni.usd
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from pxr import Gf


def randomize_color_per_instance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    color_range: tuple = (0.0, 1.0),
):
    """
    각 환경(env_ids)별로 에셋의 색상을 개별적으로 랜덤화하는 커스텀 이벤트 함수
    """
    stage = omni.usd.get_context().get_stage()
    assets = [env.scene.rigid_objects[asset_cfg.name] for asset_cfg in asset_cfgs]
    r = torch.empty(1).uniform_(color_range[0], color_range[1]).item()
    g = torch.empty(1).uniform_(color_range[0], color_range[1]).item()
    b = torch.empty(1).uniform_(color_range[0], color_range[1]).item()

    for env_id in env_ids.tolist():
        for asset in assets:
            root_path = asset.root_physx_view.prim_paths[env_id]
            color_attr_path = (
                f"{root_path}/geometry/material/Shader.inputs:diffuseColor"
            )
            color_attr = stage.GetAttributeAtPath(color_attr_path)
            if color_attr.IsValid():
                color_attr.Set(Gf.Vec3f(r, g, b))
