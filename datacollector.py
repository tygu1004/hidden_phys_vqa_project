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

        self.frame_buffer = [[] for _ in range(num_envs)]
        self.info_buffer = [{} for _ in range(num_envs)]
        self.episode_counter = 0

    def add_frame(self, observation: dict, env_indices: torch.Tensor):
        for i in env_indices:
            frame = {}
            for video_key in self.video_keys:
                frame[video_key] = observation["cams"][video_key][i]
            self.frame_buffer[i].append(frame)

    def add_info(
        self,
        env: ManagerBasedRLEnv,
        env_indices: torch.Tensor,
        obj_cfgs: list[SceneEntityCfg],
    ):
        for i in env_indices:
            if not self.info_buffer[i]:
                if self.physics_property_type == "mass":
                    for obj_cfg in obj_cfgs:
                        obj = env.scene.rigid_objects[obj_cfg.name]
                        obj_masses = obj.root_physx_view.get_masses()[i]
                        self.info_buffer[i][obj_cfg.name] = obj_masses.item()

                elif self.physics_property_type == "restitution":
                    for obj_cfg in obj_cfgs:
                        obj = env.scene.rigid_objects[obj_cfg.name]
                        meterial_properties = (
                            obj.root_physx_view.get_material_properties()
                        )
                        restitution = meterial_properties[i, :, 2]
                        self.info_buffer[i][obj_cfg.name] = restitution.item()

                else:
                    raise ValueError(
                        f"Unsupported physics property type: {self.physics_property_type}"
                    )

    def save_episode(
        self,
        env_indices: list[int],
    ):
        for env_idx in env_indices:
            if not self.frame_buffer[env_idx]:
                logging.info(
                    f"No frames collected for env index {env_idx}, skipping save."
                )
                continue

            for video_key in self.video_keys:
                video_tensor = torch.stack(
                    [frame[video_key] for frame in self.frame_buffer[env_idx]]
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
                    video_path=video_path, env_idx=env_idx
                )
                with open(self.metadata_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            self.episode_counter += 1
            self.clear_buffer([env_idx])

    def _make_metadata_per_episode(
        self,
        video_path: str,
        env_idx: int,
    ) -> str:
        bigger_key = max(self.info_buffer[env_idx], key=self.info_buffer[env_idx].get)
        if "01" in bigger_key:
            answer = "Left."
        else:
            answer = "Right."

        if self.physics_property_type == "mass":
            question = f"Looking from the front of the robot, which cube looks heavier, the left one or the right one?"

        elif self.physics_property_type == "restitution":
            question = f"Looking from the front of the robot, which sphere looks more elastic, the left or the right?"

        else:
            raise ValueError(
                f"Unsupported physics property type: {self.physics_property_type}"
            )

        return {
            "file_name": video_path,
            "question": question,
            "answer": answer,
            "values": self.info_buffer[env_idx],
        }

    def clear_buffer(self, env_indices: list[int]):
        for i in env_indices:
            self.frame_buffer[i] = []
            self.info_buffer[i] = {}
