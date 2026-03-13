import json
import logging
import os

import torch
import torchvision
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


class VQADataCollector:
    def __init__(
        self,
        repo_path: str,
        fps: int,
        num_envs: int,
        video_keys: list[str],
        physics_property_type: str = "mass",
        vcodec: str = "h264_nvenc",
    ):
        self.repo_path = repo_path
        self.fps = fps
        self.num_envs = num_envs
        self.video_keys = video_keys
        self.physics_property_type = physics_property_type
        self.vcodec = vcodec

        self.videos_dir = os.path.join(self.repo_path, "videos")
        self.metadata_path = os.path.join(self.repo_path, "metadata.jsonl")
        os.makedirs(self.videos_dir, exist_ok=True)

        self.buffer = [[] for _ in range(num_envs)]
        self.episode_counter = 0

    def add_frame(self, observation: dict, env_indices: torch.Tensor):
        for i in env_indices:
            frame = {}
            for video_key in self.video_keys:
                frame[video_key] = observation["cams"][video_key][i]
            self.buffer[i].append(frame)

    def save_episode(
        self,
        env: ManagerBasedRLEnv,
        env_indices: list[int],
        obj_cfgs: list[SceneEntityCfg],
    ):
        for env_idx in env_indices:
            if not self.buffer[env_idx]:
                logging.info(
                    f"No frames collected for env index {env_idx}, skipping save."
                )
                continue

            for video_key in self.video_keys:
                video_tensor = torch.stack(
                    [frame[video_key] for frame in self.buffer[env_idx]]
                )
                video_tensor = video_tensor.to(torch.uint8)
                video_filename = f"episode_{self.episode_counter}_{video_key}.mp4"
                video_path = os.path.join(self.videos_dir, video_filename)

                torchvision.io.write_video(
                    filename=video_path,
                    video_array=video_tensor,
                    fps=self.fps,
                    video_codec=self.vcodec,
                    options={"crf": "30"},
                )
                logging.info(f"Saved video for env index {env_idx} at {video_path}")
                metadata = self._make_metadata_per_episode(
                    video_key,
                    video_path,
                    env,
                    obj_cfgs,
                    self.physics_property_type,
                    env_idx,
                )
                with open(self.metadata_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            self.episode_counter += 1
            self.clear_buffer([env_idx])

    def _make_metadata_per_episode(
        self,
        video_key: str,
        video_path: str,
        env: ManagerBasedRLEnv,
        obj_cfgs: list[SceneEntityCfg],
        obj_type: str,
        env_idx: int,
    ) -> str:
        if len(obj_cfgs) > 2:
            raise NotImplementedError("Currently only supports only 2 objects for VQA.")

        if self.physics_property_type == "mass":
            question = f"Which {obj_type} is heavier, the left one or the right one?"
            masses = []
            for obj_cfg in obj_cfgs:
                obj = env.scene.rigid_objects[obj_cfg.name]
                obj_masses = obj.root_physx_view.get_masses()[env_idx]
                masses.append(obj_masses.item())

            if masses[0] < masses[1]:
                answer = "Left."
            else:
                answer = "Right."

            return {
                "file_name": video_path,
                "question": question,
                "answer": answer,
                "values": masses,
            }
        elif self.physics_property_type == "restitution":
            question = (
                f"Which {obj_type} is more elastic, the left one or the right one?"
            )
            restitutions = []
            for obj_cfg in obj_cfgs:
                obj = env.scene.rigid_objects[obj_cfg.name]
                meterial_properties = obj.root_physx_view.get_material_properties()
                restitution = meterial_properties[env_idx, :, 2]
                restitutions.append(restitution.item())

            if "front" in video_key:
                if restitutions[0] < restitutions[1]:
                    answer = "Left."
                else:
                    answer = "Right."
            elif "back" in video_key:
                if restitutions[0] < restitutions[1]:
                    answer = "Left."
                else:
                    answer = "Right."
            else:
                raise ValueError(
                    f"video_key should contain 'front' or 'back', but got: {video_key}"
                )

            return {
                "file_name": video_path,
                "question": question,
                "answer": answer,
                "values": restitutions,
            }
        else:
            raise ValueError(
                f"Unsupported physics property type: {self.physics_property_type}"
            )

    def clear_buffer(self, env_indices: list[int]):
        for i in env_indices:
            self.buffer[i] = []
