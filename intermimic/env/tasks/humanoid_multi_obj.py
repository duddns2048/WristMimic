# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import torch
import os
import joblib
from easydict import EasyDict
from rl_games.algos_torch import torch_ext

from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym.torch_utils import *

from env.tasks.base_task import BaseTask
from utils.torch_utils import project_to_norm
from learning.vae_loader import load_z_decoder

import time


class Humanoid_MULTI_OBJ_SMPLX(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.is_test_no_reset = self.cfg["env"].get("testNoReset", False)

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.reset_pd_action_offset =  self.cfg["args"].reset_pd_action_offset

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        # Load body configuration lists
        self.key_bodies = self.cfg["env"]["keyBodies"]
        self.hand_bodies = self.cfg["env"]["handBodies"]
        self.wrist_bodies = self.cfg["env"]["WristBodies"]
        self.key_bodies_no_elbow_no_wrist_no_shoulder = self.cfg["env"]["KeyBodies_No_Elbow_No_Wrist_No_Shoulder"]
        self.whole_bodies_no_elbow_no_hand_no_shoulder = self.cfg["env"]["WholeBodies_No_Elbow_No_Hand_No_Shoulder"]
        self.key_bodies_no_elbow_no_shoulder = self.cfg["env"]["KeyBodies_No_Elbow_No_Shoulder"]
        self.finger_bodies = self.cfg["env"]["FingerBodies"]
        self.shoulder_elbow_bodies = self.cfg["env"]["ShoulderElbowBodies"]
        self._setup_character_props()

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        self.dt = self.control_freq_inv * sim_params.dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)


        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:] = 0
        self._initial_humanoid_root_states[:, 6:7] = 1

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        # contact forces - extract humanoid contact forces only
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self._has_failed_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contactBodies"]
        self.contact_bodies = contact_bodies
        self._key_body_ids = self._build_body_ids_tensor(self.key_bodies)
        self._hand_body_ids = self._build_body_ids_tensor(self.hand_bodies)
        self._wrist_body_ids = self._build_body_ids_tensor(self.wrist_bodies)
        self._contact_body_ids = self._build_body_ids_tensor(contact_bodies)
        self._key_body_ids_no_elbow_no_wrist_no_shoulder = self._build_body_ids_tensor(self.key_bodies_no_elbow_no_wrist_no_shoulder)
        self._whole_body_ids_no_elbow_no_hand_no_shoulder = self._build_body_ids_tensor(self.whole_bodies_no_elbow_no_hand_no_shoulder)
        self._finger_body_ids = self._build_body_ids_tensor(self.finger_bodies)
        self._shoulder_elbow_body_ids = self._build_body_ids_tensor(self.shoulder_elbow_bodies)
        
        # self.initialize_vae_models()
        self.pre_physics_time=0
        self.physics_time=0
        self.post_physics_time=0

        self._refresh_sim_tensors_time=0
        self._update_hist_hoi_obs_time=0
        self._compute_hoi_observations_time=0
        self._compute_observations_time=0
        self._compute_reward_time=0
        self._compute_reset_time=0

        self.compute_observation_algo1 = 0
        self.compute_observation_algo2 = 0

        if self.viewer != None:
            self._init_camera()
            
        return

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def initialize_vae_models(self):
        self.models_path = ['checkpoints/vae/Humanoid_pulsex.pth']
        check_points = [torch_ext.load_checkpoint(ck_path) for ck_path in self.models_path]
        self.running_mean, self.running_var = \
            check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']
        self.decoder = load_z_decoder(check_points[0], activation = 'silu', z_type = 'vae', device = self.device) 

    # customize here
    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # compute latent here
        if actions.shape[-1] == 48:
            # self_obs_size = self.get_self_obs_size() # 778
            self_obs_size = 778
            self_obs = (self.obs_buf[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) \
                / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05)

            z_prior_out = self.decoder.z_prior(self_obs)
            prior_mu = self.decoder.z_prior_mu(z_prior_out)
            actions = prior_mu + actions
            actions = project_to_norm(actions, self.cfg['env'].get("embedding_norm", 5), "none")
            self_obs = torch.clamp(self_obs, min=-5.0, max=5.0)
            actions = self.decoder.decoder(torch.cat([self_obs, actions], dim = -1))

        # apply actions
        pre_physics_time_start = time.time()
        self.pre_physics_step(actions)
        pre_physics_time_end = time.time()

        # step physics and render each frame
        physics_time_start = time.time()
        self._physics_step()
        physics_time_end = time.time()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        post_physics_time_start = time.time()
        self.post_physics_step()
        post_physics_time_end = time.time()

        self.pre_physics_time+= pre_physics_time_end - pre_physics_time_start
        self.physics_time+= physics_time_end - physics_time_start
        self.post_physics_time+= post_physics_time_end - post_physics_time_start

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._has_failed_buf[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self):
        self._dof_obs_size = (51)*3
        self._num_actions = (51)*3
        self._num_obs = self.cfg["env"]["numObs"]

        return

    def get_self_obs_size(self):
        self.num_obs = 3198
        return self._num_self_obs

    def get_task_obs_size_detail(self):
        """
        Passes VAE-specific configuration from environment to network.
        This tells the network to output 48-dim latent actions instead of 153-dim.
        """
        from collections import OrderedDict
        task_obs_detail = OrderedDict()

        # Add VAE-related parameters if using latent actions
        if self.cfg['env'].get("z_type", None) == "vae":
            task_obs_detail['embedding_size'] = self.cfg['env'].get("embedding_size", 48)
            task_obs_detail['embedding_norm'] = self.cfg['env'].get("embedding_norm", 5)
            task_obs_detail['z_type'] = self.cfg['env'].get("z_type", "vae")

        return task_obs_detail

    def _build_termination_heights(self):
        self._termination_heights = 0.3
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def get_num_amp_obs(self):
        return self.ref_hoi_obs_size

    def _load_amass_gender_betas(self):
        if self._has_mesh:
            if self.humanoid_type in ['smplx']:
                gender_betas_data = joblib.load("InterAct/grab_examples/amass_isaac_gender_betas.pkl")
                self._amass_gender_betas = np.array(list(gender_betas_data.values()))
            # elif self.humanoid_type in ['smplx']:
            #     gender_betas_data = joblib.load("data/amass_x/amassx_gender_betas.pkl")
            #     self._amass_gender_betas = np.array(gender_betas_data)
            
        else:
            gender_betas_data = joblib.load("sample_data/amass_isaac_gender_betas_unique.pkl")
            self._amass_gender_betas = np.array(gender_betas_data)
          

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.robot_type

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)
        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # max_agg_bodies = self.num_humanoid_bodies + 2
        # max_agg_shapes = self.num_humanoid_shapes + 65        
        max_agg_bodies = self.num_humanoid_bodies + 200
        max_agg_shapes = self.num_humanoid_shapes + 200       
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self._build_env(i, env_ptr, humanoid_asset)
            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.robot_type
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

        # CRITICAL FIX: Prevent humanoid self-collision by setting collision filters on shapes
        # When col_filter=0, Isaac Gym allows rigid bodies within the same actor to collide
        # We add a high bit (bit 30) to all humanoid shapes so they don't collide with each other
        # but still collide with objects (which don't have bit 30)
        humanoid_props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        for shape_idx in range(len(humanoid_props)):
            # Set bit 30 for all humanoid shapes - they will ignore each other but collide with objects
            humanoid_props[shape_idx].filter = col_filter | (1 << 30)
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, humanoid_props)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.75, 0.54, 0.3))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
        self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        if self.reset_pd_action_offset:
            self._pd_action_scale[self._pd_action_offset.abs() > 0.1] = self._pd_action_scale[0].clone()
            self._pd_action_offset[self._pd_action_offset.abs() > 0.1] = 0
            self._L_knee_dof_idx = 5
            self._R_knee_dof_idx = 17
            self._pd_action_scale[self._L_knee_dof_idx] = 5
            self._pd_action_scale[self._R_knee_dof_idx] = 5
        return

    def _get_humanoid_collision_filter(self):
        # Default: return 1 (standard collision filtering)
        # This can be overridden in subclasses for selective collision
        return 1

    def _compute_reward(self, actions):
        return
    
    def _compute_reset(self):
        reset, terminated = self.compute_humanoid_reset(self.reset_buf, self.progress_buf, self.obs_buf,
                                                        self._rigid_body_pos, self.max_episode_length[self.data_id],
                                                        self._enable_early_termination, self._termination_heights, self.start_times,
                                                        self.rollout_length
                                                        )
        self._apply_reset(reset, terminated)
        return

    def _apply_reset(self, reset, terminated):
        """Write reset/terminate results into the buffers, honoring test_no_reset.

        With testNoReset enabled the episode is never cut short by a termination:
        the failure flag is accumulated (sticky for the whole episode) and the env
        is only reset once the reference sequence runs out. Subclasses should route
        their compute_*_reset results through here instead of assigning the buffers
        directly, so the behavior is shared across tasks.
        """
        if self.is_test_no_reset:
            # Accumulate fail flag (once failed, stays failed for the episode)
            self._has_failed_buf = torch.where(terminated.bool(),
                                               torch.ones_like(self._has_failed_buf),
                                               self._has_failed_buf)
            # Only reset at sequence end
            self.reset_buf[:] = (self.progress_buf >= self.max_episode_length[self.data_id] - 1).long()
            # Pass accumulated fail info as terminate
            self._terminate_buf[:] = self._has_failed_buf
        else:
            self.reset_buf[:] = reset
            self._terminate_buf[:] = terminated
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return


    def _compute_task_obs(self):
        return

    def _compute_humanoid_obs(self, env_ids=None, ref_obs=None, next_ts=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
        obs = self.compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs, self._root_height_obs,
                                                     contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids)

        return obs

    def _compute_object_obs(self, env_ids=None):
        return

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids] 
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids] 
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)       

        return

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)

        post_physics_time1 = time.time()
        self._refresh_sim_tensors()
        post_physics_time2 = time.time()
        self._update_hist_hoi_obs()
        post_physics_time3 = time.time()
        self._compute_hoi_observations(env_ids)
        post_physics_time4 = time.time()
        self._compute_observations(env_ids) # most heavy
        post_physics_time5 = time.time()
        self._compute_reward(env_ids)
        post_physics_time6 = time.time()
        self._compute_reset()
        post_physics_time7 = time.time()
        
        self._refresh_sim_tensors_time+= post_physics_time2-post_physics_time1
        self._update_hist_hoi_obs_time+= post_physics_time3-post_physics_time2
        self._compute_hoi_observations_time+= post_physics_time4-post_physics_time3
        self._compute_observations_time+= post_physics_time5-post_physics_time4
        self._compute_reward_time+= post_physics_time6-post_physics_time5
        self._compute_reset_time+= post_physics_time7-post_physics_time6

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer:
            self.gym.clear_lines(self.viewer)
            if self.debug_viz:
                self._update_debug_viz()

        return

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()

        super().render(sync_frame_time)
        return

    def _build_body_ids_tensor(self, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        return
    
    def _update_debug_viz(self):
        """Visualize wrist joint positions from simulation and reference motion."""
        self.gym.clear_lines(self.viewer)

        # key body 21 : ['L_Hip','L_Knee','L_Ankle','L_Toe','R_Hip','R_Knee','R_Ankle','R_Toe','Torso','Spine','Chest','Neck','Head',
        #               'L_Thorax','L_Shoulder','L_Elbow','L_Wrist','R_Thorax','R_Shoulder','R_Elbow','R_Wrist', 'L_Ankle','L_Toe']
        vis_part = ['L_Shoulder','L_Elbow', 'L_Wrist', 'R_Shoulder','R_Elbow','R_Wrist', 'L_Ankle','L_Toe', 'R_Ankle','R_Toe']
        object_vis = True
        # Get wrist positions from simulation
        vis_part_ids = self._build_body_ids_tensor(vis_part)
        sim_joint_pos = self._rigid_body_pos[:, vis_part_ids, :]  # [num_envs, 2, 3]

        # Get wrist positions from reference observation
        ref_joint_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs) \
            .view(self._curr_ref_obs.shape[0], -1, 3)[:, vis_part_ids]  # [num_envs, 2, 3]

        # Visualize first environment only to avoid clutter
        env_id = 0
        if env_id >= self.num_envs:
            return

        # env_ptr = self.envs[env_id]

        for env_id in range(self.num_envs):
            env_ptr = self.envs[env_id]

            # Visualization parameters
            sphere_radius = 0.03

            for wrist_idx in range(sim_joint_pos.shape[1]):
                # Simulation wrist: red sphere
                sp = sim_joint_pos[env_id, wrist_idx].cpu().numpy()
                sim_pose = gymapi.Transform()
                sim_pose.p = gymapi.Vec3(float(sp[0]), float(sp[1]), float(sp[2]))
                sim_sphere = gymutil.WireframeSphereGeometry(
                    radius=sphere_radius,
                    num_lats=6,
                    num_lons=6,
                    color=(1., 0., 0.)
                )
                gymutil.draw_lines(sim_sphere, self.gym, self.viewer, env_ptr, sim_pose)

                # Reference wrist: green sphere
                rp = ref_joint_pos[env_id, wrist_idx].cpu().numpy()
                ref_pose = gymapi.Transform()
                ref_pose.p = gymapi.Vec3(float(rp[0]), float(rp[1]), float(rp[2]))
                ref_sphere = gymutil.WireframeSphereGeometry(
                    radius=sphere_radius,
                    num_lats=6,
                    num_lons=6,
                    color=(0., 1., 0.)
                )
                gymutil.draw_lines(ref_sphere, self.gym, self.viewer, env_ptr, ref_pose)

            # --- Object position visualization ---
            if object_vis:
                sim_obj_pos = self._target_states[env_id, 0, 0:3].cpu().numpy()
                ref_obj_pos = self.extract_data_component('obj_pos', obs=self._curr_ref_obs)[env_id].cpu().numpy()

                obj_sphere_radius = 0.05

                # Sim object: red sphere
                sim_obj_pose = gymapi.Transform()
                sim_obj_pose.p = gymapi.Vec3(float(sim_obj_pos[0]), float(sim_obj_pos[1]), float(sim_obj_pos[2]))
                sim_obj_sphere = gymutil.WireframeSphereGeometry(
                    radius=obj_sphere_radius,
                    num_lats=8,
                    num_lons=8,
                    color=(1., 0., 0.)
                )
                gymutil.draw_lines(sim_obj_sphere, self.gym, self.viewer, env_ptr, sim_obj_pose)

                # Ref object: green sphere
                ref_obj_pose = gymapi.Transform()
                ref_obj_pose.p = gymapi.Vec3(float(ref_obj_pos[0]), float(ref_obj_pos[1]), float(ref_obj_pos[2]))
                ref_obj_sphere = gymutil.WireframeSphereGeometry(
                    radius=obj_sphere_radius,
                    num_lats=8,
                    num_lons=8,
                    color=(0., 1., 0.)
                )
                gymutil.draw_lines(ref_obj_sphere, self.gym, self.viewer, env_ptr, ref_obj_pose)

    def compute_humanoid_reset(self, reset_buf, progress_buf, obs_buf, rigid_body_pos,
                               max_episode_length, enable_early_termination, termination_heights, 
                               start_times, rollout_length):
        terminated = torch.zeros_like(reset_buf)

        body_height = rigid_body_pos[:, 0, 2] # root height
        body_fall = body_height < termination_heights # [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)
        # DEBUG: Track fallen humanoids

        invalid_obs = ~torch.isfinite(obs_buf)  # True where obs is NaN or infinite
        invalid_batches = torch.any(invalid_obs, dim=1)  # Check if any invalid number in each batch (B, N)
        if torch.any(invalid_obs):
            print("invalid observation")
            raise Exception("invalid observation")
        terminated = torch.where(torch.logical_or(invalid_batches, has_failed), torch.ones_like(reset_buf), terminated)
        reset = torch.where(torch.logical_or(progress_buf >= max_episode_length-1, progress_buf - start_times >= rollout_length-1), torch.ones_like(reset_buf), terminated)
        if not enable_early_termination:
            terminated = torch.where(invalid_batches, torch.ones_like(reset_buf), terminated)
        
        return reset, terminated

    # These three are added / from omnigrasp




