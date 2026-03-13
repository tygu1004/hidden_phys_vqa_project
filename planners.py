import random

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from goals import (
    get_dropping_ee_goals,
    get_pushing_ee_goals,
    get_stacking_ee_goals,
    is_stacked,
)


class TaskPlanner:
    """
    Manages the high-level logic for the pushing and stacking task.
    Handles goal generation, phase transitions (Push -> Stack), and progress tracking.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        obj_cfgs: list[SceneEntityCfg],
        device: str,
        eps: float = 0.05,
        phase_names: list[str] = ["push", "stack"],
        stabilize_steps: int = 10,
        hold_steps: int = 5,
    ):
        self.env = env
        self.obj_cfgs = obj_cfgs
        self.device = device
        self.num_envs = env.num_envs

        self.current_phase_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )
        self.current_goal_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )
        self.stabilize_countdown = torch.full(
            (self.num_envs,),
            stabilize_steps,
            device=device,
            dtype=torch.long,
        )
        self.hold_countdown = torch.full(
            (self.num_envs,), hold_steps, device=device, dtype=torch.long
        )
        self.env_ids = torch.arange(self.num_envs, device=device)

        self.phase_names = phase_names
        self.phase_goal_buffers: list[torch.Tensor] = []

        self.eps = eps
        self.stabilize_steps = stabilize_steps
        self.hold_steps = hold_steps

        for phase in self.phase_names:
            goals = self._compute_goals(phase)
            self.phase_goal_buffers.append(torch.zeros_like(goals))

    def _compute_goals(self, phase_name: str) -> torch.Tensor:
        """Computes goals based on the phase name."""
        if phase_name == "push":
            return get_pushing_ee_goals(env=self.env, obj_cfgs=self.obj_cfgs)
        elif phase_name == "stack":
            # Note: get_stacking_ee_goals depends on the order of obj_cfgs (upper/under)
            return get_stacking_ee_goals(
                env=self.env,
                upper_obj_cfg=self.obj_cfgs[0],
                under_obj_cfg=self.obj_cfgs[1],
            )
        elif phase_name == "drop":
            return get_dropping_ee_goals(
                env=self.env, obj_cfgs=self.obj_cfgs, height=0.3
            )
        else:
            raise ValueError(f"Unknown phase: {phase_name}")

    def update_stabilize_status(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates stabilization counters and returns status masks."""
        is_stabilizing = self.stabilize_countdown > 0
        just_finished = self.stabilize_countdown == 1
        self.stabilize_countdown[is_stabilizing] -= 1
        return is_stabilizing, just_finished

    def update_goals_if_needed(self, just_finished_stabilizing: torch.Tensor):
        """Generates new pushing goals for environments that just finished stabilizing."""
        if just_finished_stabilizing.any():
            target_envs_idx = just_finished_stabilizing.nonzero().flatten()

            # Reset to the first phase
            self.current_phase_idx[target_envs_idx] = 0
            self.current_goal_idx[target_envs_idx] = 0

            # Generate goals for the first phase
            self._update_goal_buffer(0, target_envs_idx)

    def _update_goal_buffer(self, phase_idx: int, env_idxs: torch.Tensor):
        """Regenerates goals for a specific phase and set of environments."""
        phase_name = self.phase_names[phase_idx]

        # Shuffle objects to randomize roles (e.g., which object is upper/under or push order)
        self.obj_cfgs = random.sample(self.obj_cfgs, len(self.obj_cfgs))

        new_goals = self._compute_goals(phase_name)
        self.phase_goal_buffers[phase_idx][:, env_idxs] = new_goals[:, env_idxs]

    def get_current_targets(self) -> torch.Tensor:
        """Returns the current target pose for each environment."""
        current_targets = torch.zeros((self.num_envs, 8), device=self.device)

        for phase_idx, goals in enumerate(self.phase_goal_buffers):
            mask = self.current_phase_idx == phase_idx
            if mask.any():
                num_steps = goals.size(dim=0)
                step_indices = self.current_goal_idx.clamp(max=num_steps - 1)

                # Select goals for the current step for all envs, then mask
                phase_targets = goals[step_indices, self.env_ids]
                current_targets = torch.where(
                    mask.unsqueeze(-1), phase_targets, current_targets
                )
        return current_targets

    def check_reach_and_advance(
        self, current_ee_pos: torch.Tensor, target_pos: torch.Tensor
    ):
        """Checks if targets are reached, updates hold counters, and advances steps."""
        dist = torch.norm(current_ee_pos - target_pos, dim=1)
        is_reached = dist < self.eps

        self.hold_countdown[is_reached] -= 1
        is_holding = self.hold_countdown > 0
        is_reached_and_not_holding = torch.logical_and(is_reached, ~is_holding)

        self.current_goal_idx[is_reached_and_not_holding] += 1
        self.hold_countdown[is_reached_and_not_holding] = self.hold_steps

        self._handle_phase_transitions()

    def _handle_phase_transitions(self):
        """Handles switching between sub-tasks (phases)."""
        num_phases = len(self.phase_names)

        for i in range(num_phases - 1):
            goals = self.phase_goal_buffers[i]
            num_steps = goals.shape[0]

            finished_mask = torch.logical_and(
                self.current_phase_idx == i, self.current_goal_idx >= num_steps
            )

            if finished_mask.any():
                transition_idxs = finished_mask.nonzero().flatten()

                next_phase = i + 1
                self.current_phase_idx[transition_idxs] = next_phase
                self.current_goal_idx[transition_idxs] = 0

                self._update_goal_buffer(next_phase, transition_idxs)

    def check_success(self) -> torch.Tensor:
        return is_stacked(self.env, self.obj_cfgs[0], self.obj_cfgs[1])

    def reset_envs(self, env_idxs: torch.Tensor):
        """Resets internal state for specific environments."""
        if env_idxs.numel() == 0:
            return
        self.stabilize_countdown[env_idxs] = self.stabilize_steps
        self.hold_countdown[env_idxs] = self.hold_steps
        self.current_goal_idx[env_idxs] = 0
        self.current_phase_idx[env_idxs] = 0
