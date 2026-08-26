from enum import Enum
import numpy as np
import torch
import os
import wandb

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.humanoid_multi_obj import *
import trimesh

from collections import OrderedDict
import time
from isaacgym import gymutil

class InterMimic_OMOMO(Humanoid_MULTI_OBJ_SMPLX):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        self._state_init = InterMimic_OMOMO.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self.motion_file = cfg['env']['motion_file'] # data directory
        self.play_dataset = cfg['env']['playdataset'] # False
        self.reward_weights = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages'] # False
        self.init_vel = cfg['env']['initVel'] # False
        self.ball_size = cfg['env']['ballSize'] # 1.
        self.more_rigid = cfg['env']['moreRigid'] # False
        self.rollout_length = cfg['env']['rolloutLength'] # 300
        self.psi = cfg['env'].get('physicalBufferSize', 1) # 3
        self.num_test_episodes = cfg['env'].get('numTestEpisodes', -1) # -1: unlimited

        # Observation format configuration
        self.use_new_obs_format = cfg['env'].get('useNewObsFormat', False) # true
        self.num_obs_old = cfg['env'].get('numObsOld', 3198)
        self.num_obs_new = cfg['env'].get('numObsNew', 3492)

        # List of indices - [0, 2] = objects at index 0 and 2 collide with human
        self._human_collision_objects = cfg['env']['humanObjectCollision']

        # Grasp window parameters (frames relative to contact)
        self.grasp_window_before = cfg['env']['graspWindowBefore']
        self.grasp_window_after = cfg['env']['graspWindowAfter']

        # Finger window parameters (frames relative to contact)
        self.finger_window_before = cfg['env']['fingerWindowBefore']
        self.finger_window_after = cfg['env']['fingerWindowAfter']

        # Reset window parameters (frames relative to contact)
        self.reset_window_before = cfg['env']['resetWindowBefore']
        self.reset_window_after = cfg['env']['resetWindowAfter']

        # Reset and reward transition frames (3 stages)
        self.reset_transition_frame1 = cfg['env']['resetTransitionFrame1']
        self.reset_transition_frame2 = cfg['env']['resetTransitionFrame2']

        # Wrist reset thresholds - 3 stages
        # Stage 1: contact-resetWindowBefore to contact+resetTransitionFrame1
        self.wrist_pos_reset_threshold1 = cfg['env']['wristPosResetThreshold1']
        self.wrist_rot_reset_threshold1 = cfg['env']['wristRotResetThreshold1']
        # Stage 2: contact+resetTransitionFrame1 to contact+resetTransitionFrame2
        self.wrist_pos_reset_threshold2 = cfg['env']['wristPosResetThreshold2']
        self.wrist_rot_reset_threshold2 = cfg['env']['wristRotResetThreshold2']
        # Stage 3: contact+resetTransitionFrame2 to contact+graspWindowAfter
        self.wrist_pos_reset_threshold3 = cfg['env']['wristPosResetThreshold3']
        self.wrist_rot_reset_threshold3 = cfg['env']['wristRotResetThreshold3']

        # Debug info
        expected_dims = self.num_obs_new if self.use_new_obs_format else self.num_obs_old
        print(f"[Observation Format] Using {'NEW' if self.use_new_obs_format else 'OLD'} format ({expected_dims} dims)")
        if self._human_collision_objects is not None:
            print(f"[Collision Filter] Human collides with objects at indices: {self._human_collision_objects}")
        else:
            print(f"[Collision Filter] Human collides with ALL objects (default)")
        print(f"[Grasp Window] -{self.grasp_window_before} to +{self.grasp_window_after} frames from contact")
        print(f"[Finger Window] -{self.finger_window_before} to +{self.finger_window_after} frames from contact")
        print(f"[Reset Window] -{self.reset_window_before} to +{self.grasp_window_after} frames from contact")
        print(f"[Reset Transition Frames] Stage 1→2: {self.reset_transition_frame1}, Stage 2→3: {self.reset_transition_frame2}")
        print(f"[Wrist Reset Stage 1 (contact-{self.reset_window_before} to contact+{self.reset_transition_frame1})] Pos: {self.wrist_pos_reset_threshold1}m, Rot: {self.wrist_rot_reset_threshold1} rad")
        print(f"[Wrist Reset Stage 2 (contact+{self.reset_transition_frame1} to contact+{self.reset_transition_frame2})] Pos: {self.wrist_pos_reset_threshold2}m, Rot: {self.wrist_rot_reset_threshold2} rad")
        print(f"[Wrist Reset Stage 3 (contact+{self.reset_transition_frame2} to contact+{self.grasp_window_after})] Pos: {self.wrist_pos_reset_threshold3}m, Rot: {self.wrist_rot_reset_threshold3} rad")
        motion_file = os.listdir(self.motion_file)
        self.motion_file = sorted([os.path.join(self.motion_file, data_path) for data_path in motion_file])        

        self.reward_object_name = [motion_example.split('_')[-2].split(',')[0] for motion_example in self.motion_file]
        self.spawn_objects_name = [motion_example.split('_')[-2].split(',') for motion_example in self.motion_file]
        self.all_spawn_objects_name = list(set(sum([spawn_objs for spawn_objs in self.spawn_objects_name], []))) # For whole dataset
        reward_object_name_set = sorted(list(set(self.reward_object_name)))

        self.max_objects = 1

        # actual reward object id
        self.reward_object_id = to_torch([reward_object_name_set.index(name) for name in self.reward_object_name], dtype=torch.long).cuda() # id in reward_object_name_set (self.reward_object_name) [0, 1, 3, 2, 1, 0, 0]
        self.reward_obj2motion = torch.stack([self.reward_object_id == k for k in range(len(reward_object_name_set))], dim=0) # [[T, F, F, F, F, T, T], [F, T, F, F, T, F, F],...]
        self.reward_object_name = reward_object_name_set
        self.robot_type = cfg['env']['robotType']

        self.object_density = cfg['env']['objectDensity']

        # Base observation size (human + obj0 reward object)
        self.base_hoi_obs_size = 7 + 51 * 6 + 52 * 13 + 13 + 52 * 3 + 52 + 1  # = 1211
        self.ref_hoi_obs_size = self.base_hoi_obs_size + (self.max_objects - 1) * 222

        self.num_motions = len(self.motion_file)
        self.dataset_index = to_torch([int(data_path.split('_')[-3].split('/')[-1][-1]) for data_path in self.motion_file], dtype=torch.long).cuda()

        # Create mapping from env_id to motion_file index
        # This ensures spawned objects match the motion data
        # Will be initialized after super().__init__() when self.num_envs is available
        self.env_to_motion_idx = None

        # Automatically set numObs based on observation format
        expected_obs_size = self.num_obs_new if self.use_new_obs_format else self.num_obs_old
        if cfg['env']['numObs'] != expected_obs_size:
            print(f"[Observation Format] Auto-adjusting numObs: {cfg['env']['numObs']} → {expected_obs_size}")
            cfg['env']['numObs'] = expected_obs_size

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.object_points_new = torch.zeros(len(self.reward_object_name),1024,3, device=self.device, dtype=torch.float) # object point tensor
        for i in range(len(self.reward_object_name)):
            self.object_points_new[i] = self.object_points[self.reward_object_name[i]]

        # Validate observation size was set correctly
        if self._num_obs != expected_obs_size:
            raise ValueError(f"[Observation Format] Internal error: numObs={self._num_obs} != expected={expected_obs_size}")

        self.hoi_data, self.hoi_refs = self._load_motion(self.motion_file, topk=self.psi)

        # Initialize env_id to motion_file index mapping
        # Each environment is assigned to a motion file in round-robin fashion
        # This matches the object spawning logic in _build_target (line 522)
        self.env_to_motion_idx = to_torch([i % self.num_motions for i in range(self.num_envs)],
                                          device=self.device, dtype=torch.long)

        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.kinematic_reset = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.contact_reset = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.dataset_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.hand_close_flag = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.hand_close_frame_count = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)

        # Store first contact frame from reference for EACH HAND independently
        # Shape: [num_envs, 2] where [:, 0] = left hand, [:, 1] = right hand
        # Value: first contact frame for that hand, or -1 if no contact in entire sequence
        self.ref_contact_frame_dual = torch.full([self.num_envs, 2], -1, device=self.device, dtype=torch.long)

        self._build_contact_body_groups()

        self._build_contact_case_body_indices()

        self._curr_reward = torch.zeros([self.num_envs, cfg['env']['rolloutLength']], device=self.device, dtype=torch.float)
        self._sum_reward = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._curr_state = torch.zeros([self.num_envs, cfg['env']['rolloutLength'], 332], device=self.device, dtype=torch.float)

        self._build_target_tensors()

    def post_physics_step(self):
        super().post_physics_step()

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        
    def _load_motion(self, motion_file, startk=0, topk=1, initk=0):
        hoi_datas = []
        hoi_refs = []

        if type(motion_file) != type([]):
            motion_file = [motion_file]
        max_episode_length = []
        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            objects = data_path.split('_')[-2]
            reward_object = data_path.split('_')[-2]

            hoi_data = torch.load(data_path)[startk:]
            loaded_dict['hoi_data'] = hoi_data.detach().to('cuda')

            max_episode_length.append(loaded_dict['hoi_data'].shape[0])
            self.fps_data = 30.

            # ============ humanoid pos & rot ===============
            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone() # [frame, 3]
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data # [frame-1, 3]
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])).to('cuda'),loaded_dict['root_pos_vel']),dim=0) # [frame, 3]

            loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7].clone() # [frame, 4]
            root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['root_rot']) # [frame, 3]
            loaded_dict['root_rot_vel'] = (root_rot_exp_map[1:,:].clone() - root_rot_exp_map[:-1,:].clone())*self.fps_data # [frame-1, 3]
            loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to('cuda'),loaded_dict['root_rot_vel']),dim=0) # [frame, 3]

            loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+153].clone() # [frame, 153]
            loaded_dict['dof_vel'] = (loaded_dict['dof_pos'][1:,:].clone() - loaded_dict['dof_pos'][:-1,:].clone())*self.fps_data # [frame-1, 153]
            loaded_dict['dof_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_vel'].shape[-1])).to('cuda'),loaded_dict['dof_vel']),dim=0) # [frame, 153]

            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 162: 162+52*3].clone() # [frame, 156]
            loaded_dict['body_pos_vel'] = (loaded_dict['body_pos'][1:,:].clone() - loaded_dict['body_pos'][:-1,:].clone())*self.fps_data # [frame-1, 156]
            loaded_dict['body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_pos_vel'].shape[-1])).to('cuda'),loaded_dict['body_pos_vel']),dim=0) # [frame, 156]

            loaded_dict['body_rot'] = loaded_dict['hoi_data'][:, 331+52:331+52+52*4].clone() # [frame, 208]
            human_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['body_rot'].view(-1, 4)).view(-1, 52*3) # [frame, 156]
            loaded_dict['body_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data # [frame-1, 156]
            loaded_dict['body_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_rot_vel'].shape[-1])).to('cuda'),loaded_dict['body_rot_vel']),dim=0) # [frame, 156]

            # ============ obj pos & rot ===============
            loaded_dict[f'obj_pos'] = loaded_dict['hoi_data'][:, 318:321].clone() # [frame, 3]
            loaded_dict[f'obj_pos_vel'] = (loaded_dict[f'obj_pos'][1:,:].clone() - loaded_dict[f'obj_pos'][:-1,:].clone())*self.fps_data # [frame-1, 3]
            if self.init_vel:
                loaded_dict[f'obj_pos_vel'] = torch.cat((loaded_dict[f'obj_pos_vel'][:1],loaded_dict[f'obj_pos_vel']),dim=0) # [frame, 3] -> initial velocity same as frame 2 velocity
            else:
                loaded_dict[f'obj_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict[f'obj_pos_vel'].shape[-1])).to('cuda'),loaded_dict[f'obj_pos_vel']),dim=0) # [frame, 3]

            loaded_dict[f'obj_rot'] = loaded_dict['hoi_data'][:, 321:325].clone() # [frame, 4]
            obj_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict[f'obj_rot']) # [frame, 3]
            loaded_dict[f'obj_rot_vel'] = (obj_rot_exp_map[1:,:].clone() - obj_rot_exp_map[:-1,:].clone())*self.fps_data # [frame-1, 3]
            loaded_dict[f'obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict[f'obj_rot_vel'].shape[-1])).to('cuda'),loaded_dict[f'obj_rot_vel']),dim=0) # [frame, 3]

            # ============ human contact ===============
            # Human contact: map values 1,2 → 1, keep -1 and map others to 0
            # Original: -1 (far), 0 (0.03-0.5m), 1 (< 0.005m), 2 (0.005-0.01m), 3-6 (intermediate)
            # Transform: -1 → -1, (1 or 2) → 1, all others (0,3,4,5,6) → 0
            human_contact_raw = loaded_dict['hoi_data'][:, 331:331+52].clone()
            if objects == reward_object:
                loaded_dict['hoi_data'][:, 331:331+52] = torch.where(
                    human_contact_raw == -1,
                    torch.tensor(-1, device=loaded_dict['hoi_data'].device),
                    torch.where(
                        (human_contact_raw == 1) | (human_contact_raw == 2),
                        torch.tensor(1, device=loaded_dict['hoi_data'].device),
                        torch.tensor(0, device=loaded_dict['hoi_data'].device)
                    )
                )
                human_contact_raw = loaded_dict['hoi_data'][:, 331:331+52].clone()

            # ============ obj contact ===============
            # Object contact: map values 1,2 → 1, all others → 0
            # Original values: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            # Transform: (1.0 or 2.0) → 1, all others (0,3,4,5,6) → 0
            obj_contact_raw = loaded_dict['hoi_data'][:, 330:331].clone()
            if objects == reward_object:
                loaded_dict['hoi_data'][:, 330:331] = torch.where(
                    (obj_contact_raw == 1.0) | (obj_contact_raw == 2.0),
                    torch.tensor(1.0, device=loaded_dict['hoi_data'].device),
                    torch.tensor(0.0, device=loaded_dict['hoi_data'].device)
                )
                obj_contact_raw = loaded_dict['hoi_data'][:, 330:331].clone()

            loaded_dict[f'obj_contact_obj'] = obj_contact_raw.clone()
            loaded_dict[f'obj_contact_human'] = human_contact_raw.clone()

            # ============ Interaction Graph (ig) ===============
            obj_rot_extend = loaded_dict[f'obj_rot'].unsqueeze(1).repeat(1, self.object_points[objects].shape[0], 1).view(-1, 4)
            object_points_extend = self.object_points[objects].unsqueeze(0).repeat(loaded_dict[f'obj_rot'].shape[0], 1, 1).view(-1, 3)
            obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict[f'obj_rot'].shape[0], self.object_points[objects].shape[0], 3) + loaded_dict[f'obj_pos'].unsqueeze(1)

            ref_ig = compute_sdf(loaded_dict['body_pos'].view(max_episode_length[-1],52,3), obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, loaded_dict['body_pos'].shape[1] // 3, 1).view(-1, 4)
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict[f'obj_rot'].shape[0], -1) # frame, 156
            loaded_dict[f'obj_ig'] = ref_ig


            # Single object reward (obj0 - reward object)
            loaded_dict['hoi_data'] = torch.cat((
                loaded_dict['root_pos'].clone(),              # f, 3
                loaded_dict['root_rot'].clone(),              # f, 4
                loaded_dict['dof_pos'].clone(),               # f, 153
                loaded_dict['dof_vel'].clone(),               # f, 153
                loaded_dict['body_pos'].clone(),              # f, 156
                loaded_dict['body_rot'].clone(),              # f, 208
                loaded_dict['body_pos_vel'].clone(),          # f, 156
                loaded_dict['body_rot_vel'].clone(),          # f, 156
                loaded_dict[f'obj_pos'].clone(),             # f, 3
                loaded_dict[f'obj_rot'].clone(),             # f, 4
                loaded_dict[f'obj_pos_vel'].clone(),         # f, 3
                loaded_dict[f'obj_rot_vel'].clone(),         # f, 3
                loaded_dict[f'obj_ig'].clone(),              # f, 156
                loaded_dict[f'obj_contact_human'].clone(),   # f, 52
                loaded_dict[f'obj_contact_obj'].clone(),     # f, 1
                # f, 1211
                ),
                dim=-1)

            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'][0:1] for _ in range(initk)]+[loaded_dict['hoi_data']], dim=0) # warm start
            hoi_datas.append(loaded_dict['hoi_data'])

            # Verify size: base 1211 + additional objects (excluding obj0)
            expected_size = self.base_hoi_obs_size + (self.max_objects - 1) * 222
            assert(expected_size == loaded_dict['hoi_data'].shape[-1]), \
                f"Expected {expected_size} but got {loaded_dict['hoi_data'].shape[-1]}"
            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1]), \
                f"ref_hoi_obs_size ({self.ref_hoi_obs_size}) != actual hoi_data size ({loaded_dict['hoi_data'].shape[-1]})"

            hoi_ref = torch.cat((
                loaded_dict['root_pos'].clone(), # f, 3 
                loaded_dict['root_rot'].clone(), # f, 4
                loaded_dict['root_pos_vel'].clone(), # f, 3
                loaded_dict['root_rot_vel'].clone(), # f, 3
                loaded_dict['dof_pos'].clone(), # f, 153
                loaded_dict['dof_vel'].clone(), # f, 153
                # f, 319
                loaded_dict[f'obj_pos'].clone(),
                loaded_dict[f'obj_pos_vel'].clone(),
                loaded_dict[f'obj_rot'].clone(),
                loaded_dict[f'obj_rot_vel'].clone())
                , dim=-1) # f, 13

            hoi_refs.append(hoi_ref)

        max_length = max(max_episode_length) + initk
        self.max_episode_length = to_torch(max_episode_length, dtype=torch.long) + initk
        hoi_data = []
        hoi_ref = []

        for i, (data, ref) in enumerate(zip(hoi_datas, hoi_refs)):
            hoi_data_pad_size = (0, 0, 0, max_length - data.size(0))
            hoi_ref_pad_size = (0, 0, 0, max_length - ref.size(0))
            hoi_data.append(F.pad(data, hoi_data_pad_size, "constant", 0))
            hoi_ref.append(F.pad(ref, hoi_ref_pad_size, "constant", 0))
        
        # different number of objects and joints for each motion
        hoi_data = torch.stack(hoi_data, dim=0)
        hoi_ref = torch.stack(hoi_ref, dim=0).unsqueeze(1).repeat(1, topk, 1, 1)

        self.ref_reward = torch.zeros((hoi_ref.shape[0], hoi_ref.shape[1], hoi_ref.shape[2])).to(hoi_ref.device)
        self.ref_reward[:, 0, :] = 1.0

        self.ref_index = torch.zeros((self.num_envs, )).long().to(hoi_ref.device)
        
        if not hasattr(self, 'data_component_order'):
            self.create_component_stat(loaded_dict)
        return hoi_data, hoi_ref

    def create_component_stat(self, loaded_dict):
        # Main component order for extract_data_component (obj_pos, obj_rot, etc. map to obj0)
        self.data_component_order = [
            'root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'body_pos', 'body_rot', 'body_pos_vel', 'body_rot_vel',
            'obj_pos', 'obj_rot', 'obj_pos_vel', 'obj_rot_vel', 'ig', 'contact_human', 'contact_obj'
        ]

        data_component_order_idx = [
            'root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'body_pos', 'body_rot', 'body_pos_vel', 'body_rot_vel',
            f'obj_pos', f'obj_rot', f'obj_pos_vel', f'obj_rot_vel',
            f'obj_ig', f'obj_contact_human', f'obj_contact_obj'
        ]

        # Precompute the sizes for each component (only obj0 for now)
        data_component_sizes = [
            loaded_dict[name].shape[1]
            for name in data_component_order_idx
        ]

        self.data_component_index = [sum(data_component_sizes[:i]) for i in range(len(data_component_sizes) + 1)]

        # Multi-object component indices (for weighting network)
        # obj0 is already in positions [0:1211], so multi-object section starts at 1211
        # and only contains obj1, obj2, ... (not obj0)
        multi_obj_start = self.data_component_index[-1]  # = 1211

        # Store indices for each object's multi-object data
        self.multi_obj_component_index = []
        obj_component_size = 222  # pos(3) + rot(4) + pos_vel(3) + rot_vel(3) + ig(156) + contact_human(52) + contact_obj(1)

        for obj_idx in range(self.max_objects):
            if obj_idx == 0:
                # obj0 data is in the original 1211 section, use those indices
                self.multi_obj_component_index.append({
                    'pos': (self.data_component_order.index('obj_pos'), self.data_component_order.index('obj_rot')),  # Use extract_data_component
                    'rot': (self.data_component_order.index('obj_rot'), self.data_component_order.index('obj_pos_vel')),
                    'pos_vel': (self.data_component_order.index('obj_pos_vel'), self.data_component_order.index('obj_rot_vel')),
                    'rot_vel': (self.data_component_order.index('obj_rot_vel'), self.data_component_order.index('ig')),
                    'ig': (self.data_component_order.index('ig'), self.data_component_order.index('contact_human')),
                    'contact_human': (self.data_component_order.index('contact_human'), self.data_component_order.index('contact_obj')),
                    'contact_obj': (self.data_component_order.index('contact_obj'), len(self.data_component_order)),
                })
                # Use actual indices for obj0 (already in original data)
                self.multi_obj_component_index[0] = {
                    'pos': (self.data_component_index[8], self.data_component_index[9]),    # obj_pos
                    'rot': (self.data_component_index[9], self.data_component_index[10]),   # obj_rot
                    'pos_vel': (self.data_component_index[10], self.data_component_index[11]), # obj_pos_vel
                    'rot_vel': (self.data_component_index[11], self.data_component_index[12]), # obj_rot_vel
                    'ig': (self.data_component_index[12], self.data_component_index[13]),   # ig
                    'contact_human': (self.data_component_index[13], self.data_component_index[14]), # contact_human
                    'contact_obj': (self.data_component_index[14], self.data_component_index[15]),   # contact_obj
                }
            else:
                # obj1, obj2, ... are in the multi-object section after position 1211
                start = multi_obj_start + (obj_idx - 1) * obj_component_size  # -1 because obj0 is not in this section
                self.multi_obj_component_index.append({
                    'pos': (start, start + 3),
                    'rot': (start + 3, start + 7),
                    'pos_vel': (start + 7, start + 10),
                    'rot_vel': (start + 10, start + 13),
                    'ig': (start + 13, start + 169),
                    'contact_human': (start + 169, start + 221),
                    'contact_obj': (start + 221, start + 222),
                })

        # =============================================================================================================

        self.ref_component_order = ['root_pos', 'root_rot', 'root_pos_vel', 'root_rot_vel', 'dof_pos', 'dof_vel']

        # Add object components (base)
        self.ref_component_order.extend([f'obj_pos', f'obj_pos_vel', f'obj_rot', f'obj_rot_vel'])

        ref_component_sizes = [loaded_dict[name].shape[1] for name in self.ref_component_order]
        
        self.ref_component_index = [sum(ref_component_sizes[:i]) for i in range(len(ref_component_sizes) + 1)]

    def extract_ref_component(self, var_name, data_id, ref_index, t):
        if var_name[:3] == 'obj':
            indices = [self.ref_component_order.index(var) for var in [f"{var_name.split('_')[0]}{var_name[3:]}"]]
            starts = [self.ref_component_index[index] for index in indices]
            ends = [self.ref_component_index[index + 1] for index in indices]
            return torch.stack([self.hoi_refs[data_id, ref_index, t, starts[0]:ends[0]]], dim=1)

        else:
            index = self.ref_component_order.index(var_name)
            start = self.ref_component_index[index]
            end = self.ref_component_index[index + 1]

            return self.hoi_refs[data_id, ref_index, t, start:end]

    def extract_data_component(self, var_name, ref=False, data_id=None, t=None, obs=None):
        index = self.data_component_order.index(var_name)
        
        # The number of columns to extract for this component.
        start = self.data_component_index[index]
        end = self.data_component_index[index+1]
        
        if ref and data_id is not None and t is not None:
            return self.hoi_data[data_id, t, start:end]
        
        if obs is not None:
            return obs[..., start:end]

    def get_reward_object_states(self, env_ids):
        """Extract reward object states for given environments as a batched tensor"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        reward_states = []
        for env_id in env_ids:
            reward_state = self._target_states[env_id, 0]  # Direct indexing by object
            reward_states.append(reward_state)
        
        return torch.stack(reward_states, dim=0)  # [num_envs, 13]

    def get_reward_object_points(self, env_ids):
        """Get object points for reward objects in given environments"""
        data_ids = self.data_id[env_ids]
        reward_objects_points = self.object_points_new[self.reward_object_id[data_ids]]

        return reward_objects_points

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _get_humanoid_collision_filter(self):
        """Override to support flexible collision filtering."""
        if self._human_collision_objects is None:
            # Default: all objects collide with human
            return super()._get_humanoid_collision_filter()

        # Selective collision: build bitmask for objects that should NOT collide with human
        # Get max number of objects (across all environments)
        max_objects = max(len(objs) for objs in self.spawn_objects_name)

        collision_filter = 0
        for obj_idx in range(max_objects):
            if obj_idx not in self._human_collision_objects:
                # Set bit for this object to prevent collision
                collision_filter |= (1 << obj_idx)

        return collision_filter

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        return   

    def _load_target_asset(self):
        # Change asset_root to ParaHome location
        asset_root = "intermimic/data/assets/smplx/sim_object"
        self._target_asset = {}
        self.object_points = {}
        self.object_normals = {}

        # Only load selected objects (but we need one asset per unique base object)
        loaded_base_objects = set()
        self.selected_base_objects = []  # Store ordered list for _build_target
        
        for obj_name in self.all_spawn_objects_name:
            loaded_base_objects.add(obj_name)
            self.selected_base_objects.append(obj_name)  # Store in ordered list
            asset_file = f"{obj_name}/{obj_name}.urdf"

            # Load asset
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.density = self.object_density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = 10
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000

            self._target_asset[obj_name] = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

            obj_file = f"{asset_root}/{obj_name}/{obj_name}.obj"
            mesh_obj = trimesh.load(obj_file, force='mesh')
            obj_verts = mesh_obj.vertices
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)
            object_points = to_torch(object_points - center)

            face_normals = mesh_obj.face_normals[object_faces]
            object_normals = to_torch(face_normals)

            # Handle padding if needed
            while object_points.shape[0] < 1024:
                points_to_pad = 1024 - object_points.shape[0]
                object_points = torch.cat([object_points, object_points[:points_to_pad]], dim=0)
                object_normals = torch.cat([object_normals, object_normals[:points_to_pad]], dim=0)

            obj_key = f'{obj_name}'
            self.object_points[obj_key] = object_points
            self.object_normals[obj_key] = object_normals

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        segmentation_id = 0

        spawn_objects = self.spawn_objects_name[env_id % len(self.spawn_objects_name)]

        target_handles = []
        for obj_idx, object_name in enumerate(spawn_objects):
            if self._human_collision_objects is None:
                # Default: all objects collide with human
                col_filter = 0
            else:
                # Selective collision: assign unique bit to each object
                # Objects that should collide with human get filter=0
                # Others get a unique bit (1 << obj_idx) that matches the human's filter
                if obj_idx in self._human_collision_objects:
                    col_filter = 0  # Collides with human
                else:
                    col_filter = 1 << obj_idx  # Blocks human (bit obj_idx)

            default_pose = gymapi.Transform()
            target_handle = self.gym.create_actor(
                env_ptr,
                self._target_asset[object_name],
                default_pose,
                object_name,
                col_group,
                col_filter,
                segmentation_id)

            props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
            for p_idx in range(len(props)):
                props[p_idx].restitution = 0.05
                props[p_idx].friction = 0.6
                props[p_idx].rolling_friction = 0.01
                props[p_idx].torsion_friction = 0.01
            self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)
            self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)
            target_handles.append(target_handle)


        self._target_handles.append(target_handles)

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        # [num_envs, 3, 13]
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1:2, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        offsets = torch.tensor([0]).to(self.device)
        self._tar_actor_ids = (self._tar_actor_ids[:, None] + offsets)
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]
        return

    def _reset_target(self, env_ids):
        self._target_states[env_ids, :,  :3] = self.extract_ref_component('obj_pos', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, :, 3:7] = self.extract_ref_component('obj_rot', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, :, 7:10] = self.extract_ref_component('obj_pos_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, :, 10:13] = self.extract_ref_component('obj_rot_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        return  

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)


        # Check individual tensor types before concatenation
        actor_id_tensors = []
        for i, env_id in enumerate(env_ids):
            tensor = self._tar_actor_ids[env_id.item()]
            tensor_int32 = tensor.to(torch.int32)
            actor_id_tensors.append(tensor_int32)
        
        env_ids_int32 = torch.cat(actor_id_tensors)
        
        # Check the unwrapped tensor
        unwrapped_tensor = gymtorch.unwrap_tensor(env_ids_int32)
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    unwrapped_tensor, len(env_ids_int32))
    
    def _reset_envs(self, env_ids):
        # DEBUG: Track when resets actually happen

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)

    def _reset_actors(self, env_ids):
        # human reset part
        if (self._state_init == InterMimic_OMOMO.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == InterMimic_OMOMO.StateInit.Start
              or self._state_init == InterMimic_OMOMO.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == InterMimic_OMOMO.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        # reset object
        self._reset_target(env_ids)
    # reset humanoid state
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        # Use fixed env-motion mapping to ensure spawned objects match motion data
        # Each environment has fixed objects spawned in _build_target, so motion must match
        i = self.env_to_motion_idx[env_ids]

        if (self._state_init == InterMimic_OMOMO.StateInit.Random
            or self._state_init == InterMimic_OMOMO.StateInit.Hybrid):
            motion_times = torch.cat([torch.randint(0, max(1, self.max_episode_length[i[e]]-self.rollout_length), (1,), device=self.device, dtype=torch.long) for e in range(num_envs)]) 
        elif (self._state_init == InterMimic_OMOMO.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device, dtype=torch.long)

        ref_reward = self.ref_reward[i, :, motion_times] 
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self.hand_close_flag[env_ids] = 0
        self.hand_close_frame_count[env_ids] = 0

        # Compute first contact frame for both hands independently
        self.ref_contact_frame_dual[env_ids] = self._compute_ref_contact_frame_dual(env_ids)

        # Update object codes for reset environments (parallelized)
        # Since envs can switch to different motions with different reward objects
        for motion_idx_unique in torch.unique(i):
            # Find all envs in this reset batch assigned to this motion
            env_mask_in_batch = (i == motion_idx_unique)
            affected_envs = env_ids[env_mask_in_batch]

            # Get reward object name for this motion
            reward_obj_name = f"{self.reward_object_name[self.reward_object_id[motion_idx_unique].item()]}"

            # Update all affected environments at once (vectorized assignment)
            # if reward_obj_name in self.obj_name_to_code:
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )

    def cal_cdf(self, i, e):
        rewards = self.ref_reward[i[e], :, :max(1, self.max_episode_length[i[e]]-self.rollout_length)].clone() 
        ref_reward_sum = 1 / (rewards.sum(dim=0)) 
        prob = ref_reward_sum / ref_reward_sum.sum()
        cdf = torch.cumsum(prob, 0)
        return cdf

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        # Use fixed env-motion mapping to ensure spawned objects match motion data
        # Each environment has fixed objects spawned in _build_target, so motion must match
        i = self.env_to_motion_idx[env_ids]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        motion_times = torch.cat([torch.searchsorted(self.cal_cdf(i, e), torch.rand(1).to(self.device)) if env_ids[e] not in ref_reset_ids else torch.zeros((1,), device=self.device, dtype=torch.long) for e in range(num_envs)]) 
        ref_reward = self.ref_reward[i, :, motion_times] 
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self.hand_close_flag[env_ids] = 0
        self.hand_close_frame_count[env_ids] = 0

        # Compute first contact frame for both hands independently
        self.ref_contact_frame_dual[env_ids] = self._compute_ref_contact_frame_dual(env_ids)

        # Update object codes for reset environments (parallelized)
        # Since envs can switch to different motions with different reward objects
        for motion_idx_unique in torch.unique(i):
            # Find all envs in this reset batch assigned to this motion
            env_mask_in_batch = (i == motion_idx_unique)
            affected_envs = env_ids[env_mask_in_batch]

            # Get reward object name for this motion
            reward_obj_name = f"{self.reward_object_name[self.reward_object_id[motion_idx_unique].item()]}_{self.reward_object_part[motion_idx_unique]}"

            # Update all affected environments at once (vectorized assignment)

        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

    def _compute_task_obs(self, env_ids=None, ref_obs=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            # _target_states is a list, need to extract elements for specific env_ids
            tar_states = [self._target_states[env_id] for env_id in env_ids]
            data_ids = self.data_id[env_ids]
        
        obs = self.compute_obj_observations(root_states, tar_states, ref_obs)
        return obs
        
        # body_pos: (2048, 52, 3)
        # body_rot: (2048, 52, 4)
    
    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids):
        # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        # heading rotation(rotation around z-axis), (2048, 4)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        # inverse heading rotation (inverse of heading rotation), (2048, 4)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        len_keypos = len(key_body_ids)
        # heading_rot_expand: (2048, 1, 4)
        heading_rot_expand = heading_rot.unsqueeze(-2)
        # heading_rot_expand_2: (2048, len_keypos, 4)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
        # flat_heading_rot_2: (2048 * len_keypos, 4)
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        # heading_rot_expand: (2048, 52, 4)
        heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
        # flat_heading_rot: (2048 * 52, 4)
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])
        # heading_rot_expand: (2048, 1, 4)
        heading_rot_expand = heading_rot.unsqueeze(-2)
        # heading_rot_expand_no_hand: (2048, 22, 4)
        heading_rot_expand_no_hand = heading_rot_expand.repeat((1, 22, 1))
        # flat_heading_rot_no_hand: (2048 * 22, 4)
        flat_heading_rot_no_hand = heading_rot_expand_no_hand.reshape(heading_rot_expand_no_hand.shape[0] * heading_rot_expand_no_hand.shape[1], 
                                                heading_rot_expand_no_hand.shape[2])
        # heading_inv_rot_expand: (2048, 1, 4)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])
        # heading_inv_rot_expand: (2048, 1, 4)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        # heading_inv_rot_expand_no_hand: (2048, 22, 4)
        heading_inv_rot_expand_no_hand = heading_inv_rot_expand.repeat((1, 22, 1))
        # flat_heading_inv_rot_no_hand: (2048 * 22, 4)
        flat_heading_inv_rot_no_hand = heading_inv_rot_expand_no_hand.reshape(heading_inv_rot_expand_no_hand.shape[0] * heading_inv_rot_expand_no_hand.shape[1], 
                                                heading_inv_rot_expand_no_hand.shape[2])
        
        # _ref_body_pos: (2048, 21, 3)
        _ref_body_pos = self.extract_data_component('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        # _body_pos: (2048, 21, 3)
        _body_pos = body_pos[:, key_body_ids, :]

        # diff_global_body_pos: (2048, 21, 3), (reference motion of +t step) - (current motion)
        diff_global_body_pos = _ref_body_pos - _body_pos
        # diff_local_body_pos_flat: (2048, 63) to local coordinates of human heading direction
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
        
        # local_ref_body_pos: (2048, 21, 3) local with respect to human root
        local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
        # local_ref_body_pos: (2048, 63) to local coordinates of human heading direction
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

        # root_pos_expand: (2048, 1, 3)
        root_pos_expand = root_pos.unsqueeze(-2)
        # local_body_pos: (2048, 52, 3) with respect to human root
        local_body_pos = body_pos - root_pos_expand
        # flat_local_body_pos: (2048 * 52, 3)
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        # flat_local_body_pos: (2048 * 52, 3) to local coordinates of human heading direction
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
        # local_body_pos: (2048, 156)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        # local_body_pos: (2048, 153)
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        # flat_body_rot: (2048 * 52, 4)
        flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
        # flat_local_body_rot: (2048 * 52, 4)
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        # flat_local_body_rot_obs: (2048 * 52, 6)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
        # local_body_rot_obs: (2048, 312)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
        
        # ref_body_rot: (2048, 208)
        ref_body_rot = self.extract_data_component('body_rot', obs=ref_obs)
        # ref_body_rot_no_hand: (2048, 88)
        ref_body_rot_no_hand = torch.cat((ref_body_rot[:, :18*4], ref_body_rot[:, 33*4:37*4]), dim=-1) 
        # body_rot_no_hand: (2048, 22, 4)
        body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
        # diff_global_body_rot: (2048 * 22, 4), rotation difference between reference motion and current motion
        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        # diff_local = heading_rot * diff_global * heading_inv_rot
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot_no_hand)
        # diff_local_body_rot_obs: (2048 * 22, 6)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        # diff_local_body_rot_obs: (2048, 132)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        # local_ref_body_rot: (2048 * 22, 4)
        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4))
        # local_ref_body_rot: (2048, 132)
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        # ref_body_vel: (2048, 21, 3)
        ref_body_vel = self.extract_data_component('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        # _body_vel: (2048, 21, 3)
        _body_vel = body_vel[:, key_body_ids, :]
        # diff_global_vel: (2048, 21, 3)
        diff_global_vel = ref_body_vel - _body_vel
        # diff_local_vel: (2048, 63), to local coordinates of human heading direction
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

        ref_body_ang_vel = self.extract_data_component('body_rot_vel', obs=ref_obs)
        ref_body_ang_vel_no_hand = torch.cat((ref_body_ang_vel[:, :18*3], ref_body_ang_vel[:, 33*3:37*3]), dim=-1)
        body_ang_vel_no_hand = torch.cat((body_ang_vel[:, :18], body_ang_vel[:, 33:37]), dim=1)
        diff_global_ang_vel = ref_body_ang_vel_no_hand.view(-1, 22, 3) - body_ang_vel_no_hand
        # diff_local_ang_vel: (2048, 66), to local coordinates of human heading direction
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot_no_hand, diff_global_ang_vel.view(-1, 3)).view(-1, 22 * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        # local_body_vel: (2048, 156), to local coordinates of human heading direction
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        # local_body_ang_vel: (2048, 156) to local coordinates of human heading direction
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

        # body_contact_buf: (2048, 31, 3)
        body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
        # contact: (2048, 31)
        contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
        # ref_body_contact: (2048, 31)
        ref_body_contact = self.extract_data_component('contact_human', obs=ref_obs)[:, contact_body_ids]
        # ref_body_contact: (2048, 31) / Only check when reference contact exists
        diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)
        obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
        return obs

    def compute_obj_observations(self, root_states, tar_states, ref_obs):
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        # Extract reward object states from the multi-object tar_states
        # tar_states is now a list of tensors, we need to get only reward objects
        if isinstance(tar_states, list):
            # Extract reward object for each environment
            env_ids = torch.arange(len(tar_states), device=self.device, dtype=torch.long)
            reward_tar_states = self.get_reward_object_states(env_ids)  # [num_envs, 13]
        else:
            # Backward compatibility for single object format
            reward_tar_states = tar_states

        tar_pos = reward_tar_states[:, 0:3]
        tar_rot = reward_tar_states[:, 3:7]
        tar_vel = reward_tar_states[:, 7:10]
        tar_ang_vel = reward_tar_states[:, 10:13]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        local_tar_pos = tar_pos - root_pos
        local_tar_pos[..., -1] = tar_pos[..., -1]
        local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
        local_tar_vel = quat_rotate(heading_rot, tar_vel)
        local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

        local_tar_rot = quat_mul(heading_rot, tar_rot)
        local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

        _ref_obj_pos = self.extract_data_component('obj_pos', obs=ref_obs)
        diff_global_obj_pos = _ref_obj_pos - tar_pos
        diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

        local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
        local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)

        ref_obj_rot = self.extract_data_component('obj_rot', obs=ref_obs)
        diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
        diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
        diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

        local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
        local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)

        ref_obj_vel = self.extract_data_component('obj_pos_vel', obs=ref_obs)
        diff_global_vel = ref_obj_vel - tar_vel
        diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

        ref_obj_ang_vel = self.extract_data_component('obj_rot_vel', obs=ref_obs)
        diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
        diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)
        obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)

        return obs

    def _compute_observations_iter(self, hoi_data, env_ids=None, delta_t=1):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)

        ts = self.progress_buf[env_ids].clone() 
        next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
        ref_obs = hoi_data[self.data_id[env_ids], next_ts].clone()
        obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)
        task_obs = self._compute_task_obs(env_ids, ref_obs)
        obs = torch.cat([obs, task_obs], dim=-1)    
        ig_all, ig, ref_ig = self._compute_ig_obs(env_ids, ref_obs)
        return torch.cat((obs,ig_all,ref_ig-ig),dim=-1)
        
    def _compute_ig_obs(self, env_ids, ref_obs):
        ig = self.extract_data_component('ig', obs=self._curr_obs[env_ids]).view(env_ids.shape[0], -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
        ig_all = ig_all.view(env_ids.shape[0], -1)    
        ref_ig = self.extract_data_component('ig', obs=ref_obs)
        ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)[:, self._key_body_ids, :]
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
        ref_ig = ref_ig.view(env_ids.shape[0], -1)
        return ig_all, ig, ref_ig
        
    def _compute_observations(self, env_ids=None):
        # if self.use_new_obs_format: 
        compute_observation_algo1_start= time.time()
        self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
        self.obs_buf[env_ids] = self._compute_observations_new(env_ids)
        compute_observation_algo1_end= time.time()
        # else:
        compute_observation_algo2_start= time.time()
        # if (env_ids is None):
        #     self._curr_ref_obs[:] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
        #     obs_t1 = self._compute_observations_iter(self.hoi_data, None, 1)
        #     obs_t16 = self._compute_observations_iter(self.hoi_data, None, 16)
        #     self.obs_buf[:] = torch.cat((obs_t1, obs_t16), dim=-1)
        # else:
        #     self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
        #     obs_t1 = self._compute_observations_iter(self.hoi_data, env_ids, 1)
        #     obs_t16 = self._compute_observations_iter(self.hoi_data, env_ids, 16)
        #     aaa = torch.cat((obs_t1, obs_t16), dim=-1) # 1599 * 2 = 3198

        compute_observation_algo2_end= time.time()
        self.compute_observation_algo1 += compute_observation_algo1_end - compute_observation_algo1_start
        self.compute_observation_algo2 += compute_observation_algo2_end - compute_observation_algo2_start

        # Compute weighting observation
        # self.weighting_obs_buf[env_ids] = self._compute_weighting_observations(env_ids)

    def _compute_hoi_observations(self, env_ids=None):
        # Get reward object points for all environments
        reward_object_points = self.get_reward_object_points(env_ids)
        self._curr_obs[:] = self.build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                        self._rigid_body_rot[:, 0, :],
                                                        self._rigid_body_vel[:, 0, :],
                                                        self._rigid_body_ang_vel[:, 0, :],
                                                        self._dof_pos, self._dof_vel, self._rigid_body_pos,
                                                        self._local_root_obs, self._root_height_obs, 
                                                        self._dof_obs_size, self._target_states,
                                                        self._tar_contact_forces, # (2048, 3)
                                                        self._contact_forces, # (2048, 52, 3)
                                                        reward_object_points,
                                                        self._rigid_body_rot,
                                                        self._rigid_body_vel,
                                                        self._rigid_body_ang_vel
                                                        )

    def build_hoi_observations(self, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, 
                            local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf, contact_buf, object_points, body_rot, body_vel, body_rot_vel):

        contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
        target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

        # Extract reward object states from multi-object target_states
        if isinstance(target_states, list):
            env_ids = torch.arange(len(target_states), device=self.device, dtype=torch.long)
            reward_target_states = self.get_reward_object_states(env_ids)  # [num_envs, 13]
        else:
            reward_target_states = target_states
        tar_pos = reward_target_states[:, 0, 0:3]
        tar_rot = reward_target_states[:, 0, 3:7]
        obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
        ig = compute_sdf(body_pos, obj_points).view(-1, 3)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, body_pos.shape[1], 1).view(-1, 4)
        ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1)

        # Build base observation (obj0 - reward object): 1211 dims
        obs_base = torch.cat((root_pos, root_rot, dof_pos, dof_vel,
                         body_pos.reshape(body_pos.shape[0],-1),
                         body_rot.reshape(body_rot.shape[0],-1),
                         body_vel.reshape(body_vel.shape[0],-1),
                         body_rot_vel.reshape(body_rot_vel.shape[0],-1),
                         reward_target_states[:,0,:], ig, contact, target_contact), dim=-1)

        # Append multi-object data (obj1, obj2, ...): (max_objects - 1) * 222 dims
        # For now, pad with zeros as placeholder - will be filled by weighting network later
        num_envs = obs_base.shape[0]
        multi_obj_dims = (self.max_objects - 1) * 222
        multi_obj_placeholder = torch.zeros((num_envs, multi_obj_dims), device=obs_base.device, dtype=obs_base.dtype)

        # Concatenate: 1211 + (max_objects - 1) * 222
        obs = torch.cat((obs_base, multi_obj_placeholder), dim=-1)

        return obs
    
    def _compute_reset(self):
        reset, terminated = self.compute_hoi_reset(self.reset_buf, self.progress_buf, self.obs_buf,
                                                   self._rigid_body_pos, self.max_episode_length[self.data_id],
                                                   self._enable_early_termination, self._termination_heights, self.start_times,
                                                   self.rollout_length, self.kinematic_reset, torch.any(self.contact_reset > 10, dim=-1)
                                                  )

        self._apply_reset(reset, terminated)

        # Compute explore_rate for resetting environments
        self._compute_progress_metrics()

        if self.reset_buf.sum() > 0 and self.psi > 1:
            reset_ind = (self.reset_buf == 1)
            data_id = self.data_id[reset_ind]
            max_episode_length = self.max_episode_length[data_id]
            if (max_episode_length < self.rollout_length).all():
                self._sum_reward[reset_ind] = 0
                return
            start_index, end_index = self.start_times[reset_ind], self.progress_buf[reset_ind]
            sum_reward = self._sum_reward[reset_ind].mean()
            if torch.rand(1)[0] < 0:
                self._sum_reward[reset_ind] = 0
                return
            reset_ind = torch.logical_and(reset_ind, self.max_episode_length[self.data_id] > self.rollout_length)
            if reset_ind.sum() < 0.995:
                return
            curr_reward = self._curr_reward[reset_ind]
            state = self._curr_state[reset_ind]
            # Initialize the reward tensor with zeros
            reward = torch.zeros((curr_reward.shape[0], self.hoi_refs.shape[0], self.hoi_refs.shape[2]), device=curr_reward.device)
            end_i = torch.minimum(max_episode_length, self.rollout_length + start_index)
            assert (end_index < end_i).all()

            for i in range(curr_reward.shape[0]):
                if end_index[i] > start_index[i]+30:  # Ensure the indices are valid
                    index_tensor = torch.arange(start_index[i]+10, end_index[i]-10, device=start_index.device)
                    reward[i, data_id[i], start_index[i]+10:end_index[i]-10] = ((end_index[i] - index_tensor) / (end_i[i] - index_tensor))
            adjust_reward, adjust_reward_index = reward.max(dim=0)
            for i in range(reward.shape[1]):
                if self.max_episode_length[i] < self.rollout_length:
                    continue
                for j in range(reward.shape[2]):
                    if self.max_episode_length[i] - j < self.rollout_length:
                        break
                    value, index = self.ref_reward[i, 1:, j].min(dim=0)
                    index = index + 1
                    id1 = adjust_reward_index[i, j]
                    idx = j - start_index[adjust_reward_index[i, j]]

                    if idx > 0 and idx < self.rollout_length and adjust_reward[i, j] > 0.5:
                        self.ref_reward[i, index, j] = adjust_reward[i, j]
                        self.hoi_refs[i, index, j] = state[id1, idx]
            self.ref_reward[:, 1:, :] = self.ref_reward[:, 1:, :] * (1 - 1e-5)
        return

    def compute_hoi_reset(self, reset_buf, progress_buf, obs_buf, rigid_body_pos,
                          max_episode_length, enable_early_termination, termination_heights, 
                          start_times, rollout_length, reset_ig, contact_reset):

        reset, terminated = self.compute_humanoid_reset(reset_buf, progress_buf, obs_buf, rigid_body_pos,
                                                        max_episode_length, enable_early_termination, termination_heights, 
                                                        start_times, rollout_length)

        reset_ig *= (progress_buf > 1 + start_times)
        contact_reset *= (progress_buf > 1 + start_times)
                
        terminated = torch.where(torch.logical_or(reset_ig, contact_reset), torch.ones_like(reset_buf), terminated)
        reset = torch.where(reset.bool(), torch.ones_like(reset_buf), terminated)
        return reset, terminated

    def _compute_reward(self, actions):
        rb, human_reset, key_pos, ref_key_pos, rp, rr , weight_h, wrist_reset_ = self.compute_humanoid_reward_new(self.reward_weights)
        ro, object_reset, obj_points, ref_obj_points = self.compute_obj_reward(self.reward_weights)
        rig, ig_reset = self.compute_ig_reward(self.reward_weights, key_pos, ref_key_pos, obj_points, ref_obj_points)
        rcg, contact_reset, rcg_hand = self.compute_cg_reward(self.reward_weights)
        # rgr, gbp, gbr, grwp, grwr, grfp, grfr, in_grasp_window, wrist_reset = self.compute_grasp_reward_new(self.reward_weights)

        # Apply humanoid reward only outside grasp window, grasp reward only inside
        # in_grasp_window is True for frames within the grasp window
        # rb_masked: humanoid reward outside grasp window, 1.0 inside grasp window

        # rb_masked = torch.where(in_grasp_window, torch.ones_like(rb), rb)

        self.rew_buf[:] = rb * ro * rig * rcg
        # self.rew_buf[:] = rb_masked * ro * rcg * rgr

        # Store reward components for logging
        self.extras['reward_components'] = {
            'rb': rb,
            'ro': ro,
            'rig': rig,
            'rcg': rcg,
            # 'rgr': rgr,
            # 'gbp': gbp,
            # 'gbr': gbr,
            # 'grwp': grwp,
            # 'grwr': grwr,
            # 'grfp': grfp,
            # 'grfr': grfr
        }

        # kinematic_reset = human_reset
        kinematic_reset = torch.logical_or(human_reset, object_reset)
        # Add wrist reset to kinematic reset
        kinematic_reset = torch.logical_or(kinematic_reset, wrist_reset_)
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        self.kinematic_reset = kinematic_reset
        # self.kinematic_reset = torch.logical_or(ig_reset, kinematic_reset)
        index = torch.arange(self._curr_reward.shape[0])
        self._curr_reward[index, self.progress_buf - self.start_times] = self.rew_buf
        self._sum_reward[index] += self.rew_buf
        self._curr_state[index, self.progress_buf - self.start_times, :] = torch.cat([
            self._humanoid_root_states,
            self._dof_pos,
            self._dof_vel,
            self._target_states[:,0,:],
        ], dim=1)
    
    def compute_humanoid_reward(self, w):
        # body pos reward
        len_keypos = len(self._key_body_ids)
        key_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._key_body_ids]
        ref_key_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._key_body_ids]

        # Interaction graph (ig): distance vectors from each body part to the object
        # ref_ig_norm: [batch, 52] - distance from each of 52 body parts to the object
        ref_ig = self.extract_data_component('ig', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3) # 1024, 52, 3
        ref_ig_norm = ref_ig.norm(dim=-1) # 1024, 52 -> [body part - object] distances

        # Weight for position tracking: closer to object → higher weight (more important)
        # exp(-5 * dist): dist=0 → weight=1, dist=0.5 → weight≈0.08
        weight_h = (-5 * ref_ig_norm).exp() # 여기
        weight_hp = weight_h.clone().detach()  
        ancle_toe_ids = [i for i in range(len_keypos) if 'Ankle' in self.key_bodies[i] or 'Toe' in self.key_bodies[i]]
        weight_hp[:, ancle_toe_ids] = 1 # ankle and toe weights to be 1: To stand?

        # ============= body position reward =============
        ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) * weight_hp[:, self._key_body_ids],dim=-1)
        rp = torch.exp(-ep*w['p'])

        # ============= Shoulder & Elbow position reward =============
        shoulder_elbow_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._shoulder_elbow_body_ids]
        ref_shoulder_elbow_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._shoulder_elbow_body_ids]
        se_ep = torch.mean(((ref_shoulder_elbow_pos - shoulder_elbow_pos)**2).sum(dim=-1) * weight_hp[:, self._shoulder_elbow_body_ids],dim=-1)
        se_rp = torch.exp(-se_ep*20) # 여기

        # ============= body rotation reward =============
        body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)
        ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 52)
        weight_hr = 1 - weight_h
        er = torch.mean(diff[:, :] * weight_hr, dim=-1)
        rr = torch.exp(-er*w['r'])
        
        # ============= Shoulder & Elbow rotation reward =============
        shoulder_elbow_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)[:, self._shoulder_elbow_body_ids]
        ref_shoulder_elbow_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)[:, self._shoulder_elbow_body_ids]
        se_diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_shoulder_elbow_rot.reshape(-1, 4)), shoulder_elbow_rot.reshape(-1, 4))
        se_diff_angle, se_diff_axis = torch_utils.quat_to_angle_axis(se_diff_quat_data)
        se_diff = se_diff_angle.view(-1, len(self._shoulder_elbow_body_ids))
        se_weight_hr = 1 - weight_h[:, self._shoulder_elbow_body_ids]
        se_er = torch.mean(se_diff[:, :] * se_weight_hr, dim=-1)
        se_rr = torch.exp(-se_er*3)

        # ============= body position velocity reward =============
        body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_obs)
        ref_body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)
        epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2,dim=-1)
        rpv = torch.exp(-epv*w['pv'])

        # ============= body rotation velocity reward =============
        dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_obs)
        ref_dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_ref_obs)
        erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
        rrv = torch.exp(-erv*w['rv'])

        # ============= body energy (acceleration) penalty ============= 
        hist_dof_vel = self.extract_data_component('dof_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('dof_vel', obs=self._curr_obs) - hist_dof_vel)*self.fps_data
        dof_diffacc = (local_vel.view(-1, 51*3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()
        energy = dof_diffacc.pow(2).mean(dim=-1).mul(-w['eg1']).exp()

        # ============= body reset marker =============
        # human_reset = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1) > 0.5
        max_dist = (ref_key_pos - key_pos).norm(dim=-1).max(dim=-1)[0]
        human_reset = max_dist > 0.35

        # ============= Final body reward =============
        # rb = rp*rr*rpv*rrv*energy
        rb = rp*rr*rpv*rrv*energy * se_rp * se_rr

        # Store sub-components
        self.extras['reward_human_components'] = {
            'rp': rp, 'rr': rr, 'rpv': rpv, 'rrv': rrv, 'energy': energy
        }

        return rb, human_reset, key_pos, ref_key_pos, rp, rr, weight_h

    def compute_humanoid_reward_new(self, w):
        
        current_frame = self.progress_buf
        left_contact_frame = self.ref_contact_frame_dual[:, 0]   # [num_envs]
        right_contact_frame = self.ref_contact_frame_dual[:, 1]  # [num_envs]
        left_has_contact = (left_contact_frame != -1)   # [num_envs] boolean
        right_has_contact = (right_contact_frame != -1)  # [num_envs] boolean

        left_win = self._compute_hand_windows(left_contact_frame, left_has_contact, current_frame)
        right_win = self._compute_hand_windows(right_contact_frame, right_has_contact, current_frame)

        in_grasp_window = ((left_has_contact & (current_frame >= left_win['grasp_start']) & (current_frame <= left_win['grasp_end'])) |
                          (right_has_contact & (current_frame >= right_win['grasp_start']) & (current_frame <= right_win['grasp_end'])))

        p_weight_tensor = torch.zeros(self.num_envs, 52, device=self.device, dtype=torch.float)
        r_weight_tensor = torch.zeros(self.num_envs, 52, device=self.device, dtype=torch.float)
        mask0 = ~left_win['in_contact_window']  &  right_win['in_contact_window'] # Only right hand in contact window
        mask1 =  left_win['in_contact_window']  & ~right_win['in_contact_window'] # Only left in contact window
        mask2 =  left_win['in_contact_window']  &  right_win['in_contact_window'] # Both hand in contact window
        mask3 = ~left_win['in_contact_window']  & ~right_win['in_contact_window'] # Both hand not in contact window
        for i, (mask,  active_body_id, w_p, w_r) in enumerate([
                                                                    (mask0, self._case0_key_body_ids,w['gbp'], w['gbr']),
                                                                    (mask1, self._case1_key_body_ids,w['gbp'],w['gbr']),
                                                                    (mask2, self._case2_key_body_ids,w['gbp'], w['gbr']),
                                                                    (mask3, self._case3_key_body_ids,w['p'], w['r'])]):
            if not mask.any():
                continue
            mask_indices = torch.where(mask)[0]
            p_weight_tensor[mask_indices[:, None], active_body_id] = w_p
            r_weight_tensor[mask_indices[:, None], active_body_id] = w_r
        
        p_weight_tensor = p_weight_tensor[:,self._key_body_ids]
        r_weight_tensor = r_weight_tensor[:,self._key_body_ids]

        # body pos reward
        len_keypos = len(self._key_body_ids)
        key_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._key_body_ids]
        ref_key_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._key_body_ids]

        # Interaction graph (ig): distance vectors from each body part to the object
        # ref_ig_norm: [batch, 52] - distance from each of 52 body parts to the object
        ref_ig = self.extract_data_component('ig', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3) # 1024, 52, 3
        ref_ig_norm = ref_ig.norm(dim=-1) # 1024, 52 -> [body part - object] distances

        # Weight for position tracking: closer to object → higher weight (more important)
        # exp(-5 * dist): dist=0 → weight=1, dist=0.5 → weight≈0.08
        weight_h = (-5 * ref_ig_norm).exp() # 여기
        weight_hp = weight_h.clone().detach()  
        ancle_toe_ids = [i for i in range(len_keypos) if 'Ankle' in self.key_bodies[i] or 'Toe' in self.key_bodies[i]]
        weight_hp[:, ancle_toe_ids] = 1 # ankle and toe weights to be 1: To stand?

        # ============= body position reward =============
        ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) *p_weight_tensor* weight_hp[:, self._key_body_ids],dim=-1)
        # rp = torch.exp(-ep*w['p'])
        rp = torch.exp(-ep)

        # ============= Shoulder & Elbow position reward =============
        # shoulder_elbow_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._shoulder_elbow_body_ids]
        # ref_shoulder_elbow_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._shoulder_elbow_body_ids]
        # se_ep = torch.mean(((ref_shoulder_elbow_pos - shoulder_elbow_pos)**2).sum(dim=-1) * weight_hp[:, self._shoulder_elbow_body_ids],dim=-1)
        # se_rp = torch.exp(-se_ep*20) # 여기

        # ============= body rotation reward =============
        body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)
        ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 52)
        weight_hr = 1 - weight_h
        er = torch.mean(diff[:, self._key_body_ids] * r_weight_tensor*weight_hr[:, self._key_body_ids], dim=-1)
        # rr = torch.exp(-er*w['r'])
        rr = torch.exp(-er)
        
        # ============= Shoulder & Elbow rotation reward =============
        # shoulder_elbow_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)[:, self._shoulder_elbow_body_ids]
        # ref_shoulder_elbow_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)[:, self._shoulder_elbow_body_ids]
        # se_diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_shoulder_elbow_rot.reshape(-1, 4)), shoulder_elbow_rot.reshape(-1, 4))
        # se_diff_angle, se_diff_axis = torch_utils.quat_to_angle_axis(se_diff_quat_data)
        # se_diff = se_diff_angle.view(-1, len(self._shoulder_elbow_body_ids))
        # se_weight_hr = 1 - weight_h[:, self._shoulder_elbow_body_ids]
        # se_er = torch.mean(se_diff[:, :] * se_weight_hr, dim=-1)
        # se_rr = torch.exp(-se_er*3)

        # ============= body position velocity reward =============
        body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_obs)
        ref_body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)
        epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2,dim=-1)
        rpv = torch.exp(-epv*w['pv'])

        # ============= body rotation velocity reward =============
        dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_obs)
        ref_dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_ref_obs)
        erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
        rrv = torch.exp(-erv*w['rv'])

        # ============= body energy (acceleration) penalty ============= 
        hist_dof_vel = self.extract_data_component('dof_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('dof_vel', obs=self._curr_obs) - hist_dof_vel)*self.fps_data
        dof_diffacc = (local_vel.view(-1, 51*3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()
        energy = dof_diffacc.pow(2).mean(dim=-1).mul(-w['eg1']).exp()

        # ============= body reset marker =============
        # human_reset = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1) > 0.5
        max_dist = (ref_key_pos - key_pos).norm(dim=-1).max(dim=-1)[0]
        human_reset = max_dist > 0.35

        # ============= 6. Wrist reset =============
        wrist_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._wrist_body_ids]
        ref_wrist_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._wrist_body_ids]
        wrist_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)[:, self._wrist_body_ids]
        ref_wrist_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)[:, self._wrist_body_ids]

        # Left wrist (index 0)
        left_grwp, left_grwr, left_pos_diff, left_ewr = self._compute_wrist_error_and_reward(
            wrist_pos[:, 0:1], ref_wrist_pos[:, 0:1], wrist_rot[:, 0:1], ref_wrist_rot[:, 0:1],
            left_win['in_reward_stages'], w
        )

        # Right wrist (index 1)
        right_grwp, right_grwr, right_pos_diff, right_ewr = self._compute_wrist_error_and_reward(
            wrist_pos[:, 1:2], ref_wrist_pos[:, 1:2], wrist_rot[:, 1:2], ref_wrist_rot[:, 1:2],
            right_win['in_reward_stages'], w
        )

        wrist_rwp = left_grwp * right_grwp
        wrist_rwr = left_grwr * right_grwr

        left_wrist_reset = self._compute_wrist_reset_condition(
            left_pos_diff, left_ewr, left_has_contact, left_win['reset_stages']
        )
        right_wrist_reset = self._compute_wrist_reset_condition(
            right_pos_diff, right_ewr, right_has_contact, right_win['reset_stages']
        )
        wrist_reset = left_wrist_reset | right_wrist_reset


        # ============= Final body reward =============
        rb = rp*wrist_rwp*wrist_rwr*rr*energy
        # rb = rp*rr*rpv*rrv*energy * se_rp * se_rr

        # Store sub-components
        self.extras['reward_human_components'] = {
            'rp': rp, 
            'rr': rr, 
            'wrist_rwp': wrist_rwp,
            'wrist_rwr': wrist_rwr,
            # 'rpv': rpv, 
            # 'rrv': rrv, 
            'energy': energy
        }

        return rb, human_reset, key_pos, ref_key_pos, rp, rr, weight_h, wrist_reset
    
    def compute_obj_reward(self, w):
        """
        Compute object reward only from the contact frame onward.
        Before contact, reward is 1.0 (no penalty).
        After contact, reward measures object position, rotation, velocity, and energy.
        """
        # Get current frame for each environment
        current_frame = self.progress_buf  # [num_envs]
        left_contact = self.ref_contact_frame_dual[:, 0]
        right_contact = self.ref_contact_frame_dual[:, 1]

        # Find earliest contact frame: if one hand has no contact (-1), use the other
        # If both have contact, use minimum. If neither has contact, use a large value (no activation)
        ref_contact_frame_min = torch.where(
            left_contact == -1,  # Left hand no contact
            torch.where(right_contact == -1, torch.full_like(left_contact, 999999), right_contact),  # Use right or large value
            torch.where(right_contact == -1, left_contact,  # Right hand no contact, use left
                       torch.min(left_contact, right_contact))  # Both have contact, use earliest
        )

        after_contact = (current_frame >= ref_contact_frame_min).float()  # (num_envs,)

        # object pos reward
        ref_obj_contact = self.extract_data_component('contact_obj', obs=self._curr_ref_obs)

        root_pos = self.extract_data_component('root_pos', obs=self._curr_obs)
        root_rot = self.extract_data_component('root_rot', obs=self._curr_obs)

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

        obj_pos = self.extract_data_component('obj_pos', obs=self._curr_obs)
        obj_rot = self.extract_data_component('obj_rot', obs=self._curr_obs)
        local_obj_pos = obj_pos - root_pos
        local_obj_pos[..., -1] = obj_pos[..., -1]
        local_obj_pos = quat_rotate(heading_rot, local_obj_pos)

        local_obj_rot = quat_mul(heading_rot, obj_rot)

        # Get reward object points for current environments
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        object_points = self.get_reward_object_points(env_ids)
        obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + obj_pos.unsqueeze(1)

        ref_root_pos = self.extract_data_component('root_pos', obs=self._curr_ref_obs)
        ref_root_rot = self.extract_data_component('root_rot', obs=self._curr_ref_obs)

        ref_heading_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)

        ref_obj_pos = self.extract_data_component('obj_pos', obs=self._curr_ref_obs)
        ref_obj_rot = self.extract_data_component('obj_rot', obs=self._curr_ref_obs)

        ref_local_obj_pos = ref_obj_pos - ref_root_pos
        ref_local_obj_pos[..., -1] = ref_obj_pos[..., -1]
        ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)

        ref_local_obj_rot = quat_mul(ref_heading_rot, ref_obj_rot)

        ref_obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        ref_obj_points = torch_utils.quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)

        # ============= Object position reward =============
        eop = torch.mean(((ref_local_obj_pos - local_obj_pos)**2),dim=-1)
        rop_base = torch.exp(-eop*w['op'])
        rop = rop_base * after_contact + (1 - after_contact)

        # ============= Object rotation reward =============
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_obj_rot), local_obj_rot)
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 1)

        eor = torch.mean(diff,dim=-1)
        ror_base = torch.exp(-eor*w['or'])
        ror = ror_base * after_contact + (1 - after_contact)

        # ============= Object position velocity reward =============
        obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_obs)
        ref_obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_ref_obs)
        eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
        ropv_base = torch.exp(-eopv*w['opv'])
        ropv = ropv_base * after_contact + (1 - after_contact)

        # ============= Object rotation velocity reward =============
        obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_obs)
        ref_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_ref_obs)
        eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
        rorv_base = torch.exp(-eorv*w['orv'])
        rorv = rorv_base * after_contact + (1 - after_contact)

        # ============= Object energy (acceleration) reward =============
        hist_obj_vel = self.extract_data_component('obj_pos_vel', obs=self._hist_obs)
        obj_diffacc = (self.extract_data_component('obj_pos_vel', obs=self._curr_obs) - hist_obj_vel)*self.fps_data
        obj_diffacc = obj_diffacc*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        hist_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('obj_rot_vel', obs=self._curr_obs) - hist_obj_rot_vel)*self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        obj_energy_base = ((obj_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp()) * (obj_rot_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp()))
        obj_energy = obj_energy_base * after_contact + (1 - after_contact)

        # ============= Object reset marker =============
        object_reset = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1) > 0.5

        # ============= Final object reward =============
        ro = rop * ror * ropv * rorv * obj_energy

        # Store sub-components
        self.extras['reward_object_components'] = {
            'rop': rop, 
            'ror': ror, 
            # 'ropv': ropv, 
            # 'rorv': rorv, 
            'obj_energy': obj_energy}
        return ro, object_reset, obj_points, ref_obj_points
    
    def compute_ig_reward(self, w, key_pos, ref_key_pos, obj_points, ref_obj_points):
        len_keypos = len(self._key_body_ids)
        ig = key_pos.view(-1,len_keypos,3).unsqueeze(2) - obj_points.unsqueeze(1)
        ref_ig = ref_key_pos.view(-1,len_keypos,3).unsqueeze(2) - ref_obj_points.unsqueeze(1)
        ### interaction graph reward ###
        weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.01))
        weight_1 = weight_1 / weight_1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.01))
        weight_2 = weight_2 / weight_2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        eig = ((ig - ref_ig)**2).sum(dim=-1) * (weight_1 + weight_2)  

        rig = torch.exp(-w['ig'] * (eig.sum(dim=-1).sum(dim=-1) * 0.5))

        reset_ig_1 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ref_ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig_2 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)
        return rig, reset_ig
    
    def compute_cg_reward(self, w):
        contact_thres = 0.1
        ref_human_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        human_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
        ref_obj_contact = self.extract_data_component('contact_obj', obs=self._curr_ref_obs)  # (2048, 1)
        obj_contact = self.extract_data_component('contact_obj', obs=self._curr_obs)  # (2048, 1)

        # Create masks: only True if reward object has contact
        ref_obj_has_contact = (ref_obj_contact > contact_thres)  # (2048, 1)
        obj_has_contact = (obj_contact > contact_thres)  # (2048, 1)

        # Apply masks: set human contact to 0 if reward object has no contact
        # This filters out contact with table/sink/floor
        ref_human_contact = ref_human_contact * ref_obj_has_contact  # (2048, 52)
        human_contact = human_contact * obj_has_contact  # (2048, 52)
        
        # ============= Left hand contact reward =============
        left_contact_hand_ids = list(range(17, 33))
        ref_left_contact_hand = ref_human_contact[:, left_contact_hand_ids]
        ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
        left_hand_contact = human_contact[:, left_contact_hand_ids].clone()
        left_hand_contact_any = torch.any(left_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        # ecg_left = (((ref_left_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(left_hand_contact - ref_left_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        ecg_left = (((ref_left_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(left_hand_contact - ref_left_contact_hand_any.unsqueeze(-1))).min(dim=-1)[0])
        rcg_left = 0.5 * (1 + torch.exp(-ecg_left*w['cg_hand'])) * (ref_left_contact_hand_any) + (1 - ref_left_contact_hand_any)

        # ============= Right hand contact reward =============
        right_contact_hand_ids = list(range(36, 52))
        ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
        ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
        right_hand_contact = human_contact[:, right_contact_hand_ids].clone()
        right_hand_contact_any = torch.any(right_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        # ecg_right = (((ref_right_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(right_hand_contact - ref_right_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        ecg_right = (((ref_right_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(right_hand_contact - ref_right_contact_hand_any.unsqueeze(-1))).min(dim=-1)[0])
        rcg_right = 0.5 * (1 + torch.exp(-ecg_right*w['cg_hand'])) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any)
        
        rcg_hand = rcg_left * rcg_right

        # ============= Other part contact reward =============
        # self.contact_bodies = 31 = 21 key body + 10 finger tips
        other_ids = [i for i in range(len(self.contact_bodies)) if i not in left_contact_hand_ids and i not in right_contact_hand_ids]
        ref_other_contact = ref_human_contact[:, other_ids]
        other_contact = human_contact[:, other_ids]
        ecg_other = ((torch.abs(other_contact - ref_other_contact) * (ref_other_contact > contact_thres))).mean(dim=-1)
        rcg_other = torch.exp(-ecg_other*w['cg_other'])
        
        # ============= .. reward =============
        no_contact = torch.abs(human_contact) < contact_thres
        ecg_all = (torch.abs(no_contact + ref_human_contact) * (ref_human_contact < -contact_thres)).mean(dim=-1)
        rcg_all = torch.exp(-ecg_all*w['cg_all'])

        # ============= .. .. =============
        contact_all = self._contact_forces.clone().abs().sum(dim=-1).sum(dim=-1)
        contact_energy = contact_all.pow(2).mul(-w['eg3']).exp()

        # ============= Reset condition: relative error exceeds 2x =============
        # hand contact difference only when reference hand contact exists - consider hand as one big part
        contact_reset = torch.cat([ 
                                torch.abs(ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any) * ref_left_contact_hand_any.unsqueeze(-1), 
                                torch.abs(ref_right_contact_hand_any.unsqueeze(-1) - right_hand_contact_any) * ref_right_contact_hand_any.unsqueeze(-1),
                                ], dim=-1)

        # ============= Final contact reward =============
        # rcg = rcg_hand*rcg_other*rcg_all*contact_energy
        rcg = rcg_hand*rcg_other*contact_energy
        # rcg = rcg_other

        # Store sub-components
        self.extras['reward_contact_components'] = {
            'rcg_hand': rcg_hand, 'rcg_other': rcg_other,
            'rcg_all': rcg_all, 'contact_energy': contact_energy
        }
        return rcg, contact_reset, rcg_hand

    def play_dataset_step(self, time):
        t = time
        if t == 0:
            # Use the pre-computed env_to_motion_idx mapping to ensure
            # each environment uses the motion data matching its spawned objects
            self.data_id = self.env_to_motion_idx.clone()

        env_ids = to_torch([i for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        t = to_torch(
                [
                    t if t < self.max_episode_length[self.data_id[i]] else self.max_episode_length[self.data_id[i]]-1
                    for i in range(self.num_envs)
                ],
                device=self.device,
                dtype=torch.long
            )
        
        ### update object states ###
        self._target_states[env_ids, :, :3] = self.extract_ref_component('obj_pos', self.data_id[env_ids], self.ref_index[env_ids], t)
        self._target_states[env_ids, :, 3:7] = self.extract_ref_component('obj_rot', self.data_id[env_ids], self.ref_index[env_ids], t)
        self._target_states[env_ids, :, 7:10] = torch.zeros_like(self._target_states[env_ids, :, 7:10])
        self._target_states[env_ids, :, 10:13] = torch.zeros_like(self._target_states[env_ids, :, 10:13])


        # Update humanoid states   
        _humanoid_root_pos = self.extract_data_component('root_pos', True, self.data_id[env_ids], t)
        _humanoid_root_rot = self.extract_data_component('root_rot', True, self.data_id[env_ids], t)
        _humanoid_dof_pos = self.extract_data_component('dof_pos', True, self.data_id[env_ids], t)
        _humanoid_dof_vel = self.extract_data_component('dof_vel', True, self.data_id[env_ids], t)
        
        self._humanoid_root_states[env_ids, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[env_ids, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[:, 7:10] = torch.zeros_like(self._humanoid_root_states[:, 7:10])
        self._humanoid_root_states[:, 10:13] = torch.zeros_like(self._humanoid_root_states[:, 10:13])
        self._dof_pos[env_ids] = _humanoid_dof_pos
        self._dof_vel[env_ids] = _humanoid_dof_vel
        
        env_ids_int32 = self._humanoid_actor_ids[env_ids].to(torch.int32)

        # human root
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # human dof                                       
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        env_ids_int32 = torch.cat([self._tar_actor_ids[env_id.item()].to(torch.int32) for env_id in env_ids])

        # object root        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                    gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._refresh_sim_tensors()
        obj_contact = self.extract_data_component('contact_obj', True, self.data_id[env_ids], t)
        obj_contact = torch.any(obj_contact > 0.1, dim=-1)
        human_contact = self.extract_data_component('contact_human', True, self.data_id[env_ids], t)
        for env_id, env_ptr in enumerate(self.envs):
            if env_id in env_ids:
                env_ptr = self.envs[env_id]
                handle = self._target_handles[env_id]
                handle = [handle] if isinstance(handle, int) else handle
                if obj_contact[env_id] == True:
                    self.gym.set_rigid_body_color(env_ptr, handle[0], 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(1., 0., 0.))
                else:
                    self.gym.set_rigid_body_color(env_ptr, handle[0], 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(0., 0., 1.))

                handle = self.humanoid_handles[env_id]
                # This is just for color setting
                for j in range(self.num_bodies):
                    if human_contact[env_id, j] > 0.5: # 0~3cm red
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(1., 0., 0.))
                    elif human_contact[env_id, j] > -0.5: # 3cm ~ 50cm green
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 1., 0.))
                    else: # far then 50cm blue
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 0., 1.))
        self.render(t=t)
        self.gym.simulate(self.sim)

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer and self.save_images:
            env_ids = 0
            frame_id = t if self.play_dataset else self.progress_buf[env_ids]
            dataname = self.motion_file[-1][6:-3]
            rgb_filename = f"intermimic/data/images/{dataname}/rgb_env{env_ids}_frame{frame_id:05d}.png"
            os.makedirs(f"intermimic/data/images/{dataname}", exist_ok=True)
            self.gym.write_viewer_image_to_file(self.viewer, rgb_filename)
    
    def _compute_hand_windows(self, contact_frame, has_contact, current_frame):
        """Compute grasp/finger/reset windows and stage masks for one hand."""
        # Grasp window
        contact_window_start = torch.clamp(contact_frame - self.grasp_window_before, min=0)
        contact_window_end = contact_frame + self.grasp_window_after

        # in contact window
        in_contact_window = has_contact & (current_frame >= contact_window_start) & (current_frame <= contact_window_end)
        
        # Reward stages
        approach_stage = has_contact & (current_frame >= contact_window_start) & (current_frame <= contact_frame + self.reset_transition_frame1)
        grasping_stage = has_contact & (current_frame > contact_frame + self.reset_transition_frame1) & (current_frame <= contact_frame + self.reset_transition_frame2)
        stabilization_stage = has_contact & (current_frame > contact_frame + self.reset_transition_frame2) & (current_frame <= contact_window_end)

        return {
            'grasp_start': contact_window_start, 'grasp_end': contact_window_end,
            'in_contact_window': in_contact_window,
            'in_reward_stages': (approach_stage, grasping_stage, stabilization_stage),
            'reset_stages': (approach_stage, grasping_stage, stabilization_stage),
        }

    def _compute_wrist_error_and_reward(self, wrist_pos, ref_wrist_pos, wrist_rot, ref_wrist_rot, reward_stages, w):
        """Compute wrist position/rotation error and reward for one hand."""
        # Position error
        ewp = torch.mean(((ref_wrist_pos - wrist_pos)**2).sum(dim=-1), dim=-1)

        # Rotation error
        diff_quat = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_wrist_rot.reshape(-1, 4)), wrist_rot.reshape(-1, 4))
        diff_angle, _ = torch_utils.quat_to_angle_axis(diff_quat)
        ewr = torch.mean(diff_angle.view(-1, 1), dim=-1)

        # Stage-wise rewards
        stage1, stage2, stage3 = reward_stages
        pos_weight = w['gwp1'] * stage1.float() + w['gwp2'] * stage2.float() + w['gwp3'] * stage3.float()
        rot_weight = w['gwr1'] * stage1.float() + w['gwr2'] * stage2.float() + w['gwr3'] * stage3.float()

        grwp = torch.exp(-ewp * pos_weight)
        grwr = torch.exp(-ewr * rot_weight)

        # Position diff for reset (L2 norm, not squared)
        pos_diff = torch.mean(torch.norm(ref_wrist_pos - wrist_pos, dim=-1), dim=-1)

        return grwp, grwr, pos_diff, ewr
    
    def _compute_finger_error_and_reward(self, finger_pos, ref_finger_pos, finger_rot, ref_finger_rot, finger_stages, w, num_fingers=15):
        """Compute finger position/rotation error and reward for one hand."""
        # Position error
        efp = torch.mean(((ref_finger_pos - finger_pos)**2).sum(dim=-1), dim=-1)

        # Rotation error
        diff_quat = torch_utils.quat_mul_norm(
            torch_utils.quat_inverse(ref_finger_rot.reshape(-1, 4)),
            finger_rot.reshape(-1, 4)
        )
        diff_angle, _ = torch_utils.quat_to_angle_axis(diff_quat)
        efr = torch.mean(diff_angle.view(-1, num_fingers), dim=-1)

        # Stage-wise rewards
        stage1, stage2, stage3 = finger_stages
        pos_weight = w['gfp1'] * stage1.float() + w['gfp2'] * stage2.float() + w['gfp3'] * stage3.float()
        rot_weight = w['gfr1'] * stage1.float() + w['gfr2'] * stage2.float() + w['gfr3'] * stage3.float()

        grfp = torch.exp(-efp * pos_weight)
        grfr = torch.exp(-efr * rot_weight)

        return grfp, grfr
    
    def _compute_wrist_reset_condition(self, pos_diff, rot_diff, has_contact, reset_stages):
        """Compute wrist reset condition for one hand."""
        stage1, stage2, stage3 = reset_stages

        cond1 = (pos_diff > self.wrist_pos_reset_threshold1) | (rot_diff > self.wrist_rot_reset_threshold1)
        cond2 = (pos_diff > self.wrist_pos_reset_threshold2) | (rot_diff > self.wrist_rot_reset_threshold2)
        cond3 = (pos_diff > self.wrist_pos_reset_threshold3) | (rot_diff > self.wrist_rot_reset_threshold3)

        reset1 = has_contact & cond1 & (self.progress_buf >= 10) & stage1
        reset2 = has_contact & cond2 & (self.progress_buf >= 10) & stage2
        reset3 = has_contact & cond3 & (self.progress_buf >= 10) & stage3

        return reset1 | reset2 | reset3

# =============== Grasping reward ====================

    def compute_grasp_reward_new(self, w):
        """
        Compute grasp reward with DUAL-HAND independent tracking.

        Key changes from original:
        - Removed case-based system (0/1/2/3)
        - Tracks left and right hand contact frames independently
        - Each hand has its own window computation and stage transitions
        - Wrist/finger rewards computed per-hand and multiplied
        - Reset logic uses strict OR (any hand violation triggers reset)
        """
        # Get current frame for each environment
        current_frame = self.progress_buf  # [num_envs]

        # ----------- 1. Contact info -----------
        # Extract dual contact frames: [num_envs, 2] where [:, 0] = left, [:, 1] = right
        left_contact_frame = self.ref_contact_frame_dual[:, 0]   # [num_envs]
        right_contact_frame = self.ref_contact_frame_dual[:, 1]  # [num_envs]
        left_has_contact = (left_contact_frame != -1)   # [num_envs] boolean
        right_has_contact = (right_contact_frame != -1)  # [num_envs] boolean
        
        # ============= 2. Compute windows for both hands =============
        left_win = self._compute_hand_windows(left_contact_frame, left_has_contact, current_frame)
        right_win = self._compute_hand_windows(right_contact_frame, right_has_contact, current_frame)

        # ========== UNION WINDOWS (for overall grasp/finger masking) ==========
        in_grasp_window = (left_win['in_contact_window'] | right_win['in_contact_window'])

        in_finger_window = ((left_has_contact & (current_frame >= left_win['finger_start']) & (current_frame <= left_win['finger_end'])) |
                           (right_has_contact & (current_frame >= right_win['finger_start']) & (current_frame <= right_win['finger_end'])))

        # ============= 3. Body position/rotation reward (gbp, gbr) =============
        # Use contact-based body selection (not case-based)
        # Strategy: Exclude arms that have contact, include arms that don't
        len_keypos = len(self._key_body_ids)
        ref_ig = self.extract_data_component('ig', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_ig_norm = ref_ig.norm(dim=-1)  # [num_envs, 52]
        weight_h = (-5 * ref_ig_norm).exp()

        weight_hp = weight_h.clone().detach()
        ancle_toe_ids = [i for i in range(len_keypos) if 'Ankle' in self.key_bodies[i] or 'Toe' in self.key_bodies[i]]
        weight_hp[:, ancle_toe_ids] = 1

        gbp = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        gbr = torch.ones(self.num_envs, device=self.device, dtype=torch.float)

        # Determine body indices based on which hands have contact
        # Case logic:
        # - Both hands contact → use case2 bodies (exclude both arms)
        # - Left only contact → use case1 bodies (exclude left arm, include right)
        # - Right only contact → use case0 bodies (exclude right arm, include left)
        # - No contact → use case3 bodies (include both arms)

        both_contact_mask = left_has_contact & right_has_contact # [num_envs] boolean
        left_only_mask = left_has_contact & ~right_has_contact   # [num_envs] boolean
        right_only_mask = right_has_contact & ~left_has_contact  # [num_envs] boolean
        no_contact_mask = ~left_has_contact & ~right_has_contact # [num_envs] boolean

        # Compute gbp/gbr for each combination
        for mask, key_body_ids, whole_body_ids in [
            (both_contact_mask, self._case3_key_body_ids, self._case3_whole_body_ids), # 2
            (left_only_mask, self._case3_key_body_ids, self._case3_whole_body_ids), # 1
            (right_only_mask, self._case3_key_body_ids, self._case3_whole_body_ids), # 0 
            (no_contact_mask, self._case3_key_body_ids, self._case3_whole_body_ids) # 3
        ]:
            if not mask.any():
                continue

            # Body position reward
            key_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, key_body_ids]
            ref_key_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, key_body_ids]
            ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) * weight_hp[:, key_body_ids], dim=-1)
            gbp[mask] = torch.exp(-ep[mask] * w['gbp'])

            # Body rotation reward
            body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)[:, whole_body_ids]
            ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)[:, whole_body_ids]
            diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
            diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
            diff = diff_angle.view(-1, len(whole_body_ids))
            weight_hr = 1 - weight_h
            er = torch.mean(diff[:, :] * weight_hr[:, whole_body_ids], dim=-1)
            gbr[mask] = torch.exp(-er[mask] * w['gbr'])

        # ============= 4. Wrist rewards =============
        # Extract all wrist data: [num_envs, 2, 3/4] where index 0=left, 1=right
        wrist_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._wrist_body_ids]
        ref_wrist_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._wrist_body_ids]
        wrist_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)[:, self._wrist_body_ids]
        ref_wrist_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)[:, self._wrist_body_ids]

        # Left wrist (index 0)
        left_grwp, left_grwr, left_pos_diff, left_ewr = self._compute_wrist_error_and_reward(
            wrist_pos[:, 0:1], ref_wrist_pos[:, 0:1], wrist_rot[:, 0:1], ref_wrist_rot[:, 0:1],
            left_win['in_reward_stages'], w
        )

        # Right wrist (index 1)
        right_grwp, right_grwr, right_pos_diff, right_ewr = self._compute_wrist_error_and_reward(
            wrist_pos[:, 1:2], ref_wrist_pos[:, 1:2], wrist_rot[:, 1:2], ref_wrist_rot[:, 1:2],
            right_win['in_reward_stages'], w
        )

        # COMBINE left and right wrist rewards (multiply)
        grwp = left_grwp * right_grwp
        grwr = left_grwr * right_grwr

        # ========== FINGER REWARDS (PER-HAND) ==========
        # Extract all finger data: [num_envs, 30, 3/4] where 0-14=left, 15-29=right
        finger_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._finger_body_ids]
        ref_finger_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._finger_body_ids]
        finger_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)[:, self._finger_body_ids]
        ref_finger_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)[:, self._finger_body_ids]

        # Left fingers (0:15)
        left_grfp, left_grfr = self._compute_finger_error_and_reward(
            finger_pos[:, :15], ref_finger_pos[:, :15], finger_rot[:, :15], ref_finger_rot[:, :15],
            left_win['finger_stages'], w
        )
        # Right fingers (15:30)
        right_grfp, right_grfr = self._compute_finger_error_and_reward(
            finger_pos[:, 15:], ref_finger_pos[:, 15:], finger_rot[:, 15:], ref_finger_rot[:, 15:],
            right_win['finger_stages'], w
        )

        # COMBINE left and right finger rewards (multiply)
        grfp = left_grfp * right_grfp
        grfr = left_grfr * right_grfr

        # ============= 6. Wrist reset =============
        left_wrist_reset = self._compute_wrist_reset_condition(
            left_pos_diff, left_ewr, left_has_contact, left_win['reset_stages']
        )
        right_wrist_reset = self._compute_wrist_reset_condition(
            right_pos_diff, right_ewr, right_has_contact, right_win['reset_stages']
        )
        wrist_reset = left_wrist_reset | right_wrist_reset

        # ========== FINAL REWARD COMBINATION ==========
        grasp_mask = in_grasp_window.float()
        finger_mask = in_finger_window.float()
        gr = (gbp * gbr * grwp * grwr) * grasp_mask + (1 - grasp_mask)

        # Return same signature as original
        return gr, gbp, gbr, grwp, grwr, grfp, grfr, in_grasp_window, wrist_reset


    def _compute_ref_contact_frame_dual(self, env_ids):
        """
        For each environment in env_ids, find the first contact frame for BOTH hands independently.

        Returns:
            Tensor of shape [len(env_ids), 2] where:
            [:, 0] = left hand first contact frame (-1 if no contact)
            [:, 1] = right hand first contact frame (-1 if no contact)

        Detection logic:
            - Scan contact_human data for all frames (not just contact_obj)
            - Left hand contact: any body in _left_arm_contact_ids has contact > 0.1
            - Right hand contact: any body in _right_arm_contact_ids has contact > 0.1
            - Returns -1 if no contact found (instead of max_episode_length)
        """
        contact_frames_dual = torch.full((len(env_ids), 2), -1, dtype=torch.long, device=self.device)

        for idx, env_id in enumerate(env_ids):
            data_id = self.data_id[env_id]
            max_frames = self.max_episode_length[data_id]

            # Get contact_human data for all frames: [num_frames, 52]
            # contact_human has 52 values per frame (one per SMPLX rigid body)
            contact_human_all = self.extract_data_component('contact_human', ref=True,
                                                            data_id=data_id.unsqueeze(0),
                                                            t=torch.arange(max_frames, device=self.device))
            # Shape: [num_frames, 52]

            # LEFT hand contact detection
            # Check if ANY left arm body has contact > 0.1 at each frame
            left_arm_contacts = contact_human_all[:, self._left_arm_contact_ids]  # [num_frames, 18]
            left_contact_per_frame = (left_arm_contacts > 0.1).any(dim=1)  # [num_frames]

            if left_contact_per_frame.any():
                first_left_contact = torch.where(left_contact_per_frame)[0][0]
                contact_frames_dual[idx, 0] = first_left_contact
            # else: remains -1

            # RIGHT hand contact detection
            # Check if ANY right arm body has contact > 0.1 at each frame
            right_arm_contacts = contact_human_all[:, self._right_arm_contact_ids]  # [num_frames, 18]
            right_contact_per_frame = (right_arm_contacts > 0.1).any(dim=1)  # [num_frames]

            if right_contact_per_frame.any():
                first_right_contact = torch.where(right_contact_per_frame)[0][0]
                contact_frames_dual[idx, 1] = first_right_contact
            # else: remains -1

        return contact_frames_dual

    def _compute_progress_metrics(self):
        """
        Compute progress metrics for environments that are resetting.

        contact_progress = elapsed_frames / first_contact_frame
            < 1: Reset before contact, learning body motion
            ~ 1: Reset around contact, exploring grasping
            > 1: Passed contact, found successful grasp strategy

        episode_progress = elapsed_frames / max_episode_length
            0~1: Fraction of full sequence completed before reset
            = 1: Completed full episode without early termination

        Stores individual values as tensors in extras for proper weighted averaging.
        """
        if self.reset_buf.sum() == 0:
            self.extras['contact_progress'] = None
            self.extras['episode_progress'] = None
            return

        reset_ids = torch.where(self.reset_buf == 1)[0]

        # Elapsed frames since episode start
        elapsed_frames = (self.progress_buf[reset_ids] - self.start_times[reset_ids]).float()

        # === Contact Progress ===
        # Get first contact frame (minimum of left and right hand contact)
        left_contact = self.ref_contact_frame_dual[reset_ids, 0]
        right_contact = self.ref_contact_frame_dual[reset_ids, 1]

        # Find earliest contact frame: if one hand has no contact (-1), use the other
        # If both have contact, use minimum. If neither has contact, result is -1
        first_contact = torch.where(
            left_contact == -1, right_contact,
            torch.where(right_contact == -1, left_contact,
                        torch.minimum(left_contact, right_contact))
        ).float()

        # Only compute for envs with valid contact frame (> 0)
        valid_contact_mask = first_contact > 0

        if valid_contact_mask.sum() > 0:
            valid_elapsed = elapsed_frames[valid_contact_mask]
            valid_contact = first_contact[valid_contact_mask]
            contact_progress = valid_elapsed / valid_contact
            self.extras['contact_progress'] = contact_progress  # shape: [num_valid_reset_envs]
        else:
            self.extras['contact_progress'] = None

        # === Episode Progress ===
        # Get max episode length for each resetting env
        max_ep_length = self.max_episode_length[self.data_id[reset_ids]].float()

        # All envs have valid max_episode_length
        episode_progress = elapsed_frames / max_ep_length
        self.extras['episode_progress'] = episode_progress  # shape: [num_reset_envs]

    def get_reward_object_normals(self, env_ids):
        """Get object surface normals for reward objects in given environments"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        data_ids = self.data_id[env_ids]
        reward_objects = [f"{self.reward_object_name[self.reward_object_id[data_id]]}_{self.reward_object_part[data_id]}" for data_id in data_ids]
        reward_objects_normals = torch.stack([self.object_normals[reward_object] for reward_object in reward_objects], dim=0)
        return reward_objects_normals
    # Build body group indices for contact detection
    def _build_contact_body_groups(self):
        """
        Build body group indices for right and left arm/hand for contact detection.
        Uses ALL finger joints (not just fingertips) plus wrist, elbow, shoulder.
        Right arm: [R_Shoulder, R_Elbow, R_Wrist, R_Index1-3, R_Middle1-3, R_Pinky1-3, R_Ring1-3, R_Thumb1-3]
        Left arm: [L_Shoulder, L_Elbow, L_Wrist, L_Index1-3, L_Middle1-3, L_Pinky1-3, L_Ring1-3, L_Thumb1-3]
        Note: Thorax is NOT included in arm groups.
        """
        # Right arm body names (all finger joints for comprehensive detection)
        right_arm_bodies = [
            'R_Shoulder', 'R_Elbow', 'R_Wrist',
            'R_Index1', 'R_Index2', 'R_Index3',
            'R_Middle1', 'R_Middle2', 'R_Middle3',
            'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
            'R_Ring1', 'R_Ring2', 'R_Ring3',
            'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
        ]

        # Left arm body names (all finger joints for comprehensive detection)
        left_arm_bodies = [
            'L_Shoulder', 'L_Elbow', 'L_Wrist',
            'L_Index1', 'L_Index2', 'L_Index3',
            'L_Middle1', 'L_Middle2', 'L_Middle3',
            'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
            'L_Ring1', 'L_Ring2', 'L_Ring3',
            'L_Thumb1', 'L_Thumb2', 'L_Thumb3'
        ]

        # Build indices using Isaac Gym to get actual SMPLX body indices
        # contact_human has 52 values corresponding to SMPLX rigid bodies (index 0 = Pelvis)
        # We use _build_body_ids_tensor() to get the correct SMPLX rigid body indices
        # Right arm: indices 34-51 (R_Shoulder/Elbow/Wrist + all right finger segments)
        # Left arm: indices 15-32 (L_Shoulder/Elbow/Wrist + all left finger segments)
        self._right_arm_contact_ids = self._build_body_ids_tensor(right_arm_bodies)
        self._left_arm_contact_ids = self._build_body_ids_tensor(left_arm_bodies)
    # Build 4 sets of body indices for different contact cases
    def _build_contact_case_body_indices(self):
        """
        Build 4 sets of body indices for gbp/gbr rewards based on contact cases.
        Case 0 (right hand contact): Exclude R_Wrist, R_Elbow, R_Shoulder (keep L_Wrist, L_Elbow, L_Shoulder)
        Case 1 (left hand contact): Exclude L_Wrist, L_Elbow, L_Shoulder (keep R_Wrist, R_Elbow, R_Shoulder)
        Case 2 (both hands contact): Exclude both L and R Wrist, Elbow, Shoulder
        Case 3 (no hand contact): Include both L and R Wrist, Elbow, Shoulder
        """
        # Define arm body parts
        # right_arm_parts = ['R_Wrist', 'R_Elbow', 'R_Shoulder']
        right_arm_parts = ['R_Elbow', 'R_Shoulder']
        # left_arm_parts = ['L_Wrist', 'L_Elbow', 'L_Shoulder']
        left_arm_parts = ['L_Elbow', 'L_Shoulder']
        both_arm_parts = right_arm_parts + left_arm_parts

        # For BOTH key_bodies and whole_bodies: use consistent logic
        # Start with base lists that have NO arm parts (from config)
        # Then ADD back the non-contacting arm parts
        base_key_bodies_no_arms = self.key_bodies_no_elbow_no_shoulder # wrist included
        base_whole_bodies_no_arms = self.whole_bodies_no_elbow_no_hand_no_shoulder

        # Case 0: Right hand contact - add left arm (right arm stays excluded)
        case0_key_bodies = base_key_bodies_no_arms + left_arm_parts
        case0_whole_bodies = base_whole_bodies_no_arms + left_arm_parts

        # Case 1: Left hand contact - add right arm (left arm stays excluded)
        case1_key_bodies = base_key_bodies_no_arms + right_arm_parts
        case1_whole_bodies = base_whole_bodies_no_arms + right_arm_parts

        # Case 2: Both hands contact - add neither arm (both stay excluded)
        case2_key_bodies = base_key_bodies_no_arms
        case2_whole_bodies = base_whole_bodies_no_arms

        # Case 3: No hand contact - add both arms
        case3_key_bodies = base_key_bodies_no_arms + both_arm_parts
        case3_whole_bodies = base_whole_bodies_no_arms + both_arm_parts

        # Build tensors of indices
        self._case0_key_body_ids = self._build_body_ids_tensor(case0_key_bodies)
        self._case0_whole_body_ids = self._build_body_ids_tensor(case0_whole_bodies)

        self._case1_key_body_ids = self._build_body_ids_tensor(case1_key_bodies)
        self._case1_whole_body_ids = self._build_body_ids_tensor(case1_whole_bodies)

        self._case2_key_body_ids = self._build_body_ids_tensor(case2_key_bodies)
        self._case2_whole_body_ids = self._build_body_ids_tensor(case2_whole_bodies)

        self._case3_key_body_ids = self._build_body_ids_tensor(case3_key_bodies)
        self._case3_whole_body_ids = self._build_body_ids_tensor(case3_whole_bodies)

# ========== Functions for Multi-object weighting ===============

    def _get_weighting_obs_size(self):
        """Define size of weighting observation from config"""
        return self.cfg["env"].get("numWeightingObs", 30)

    def _compute_weighting_observations(self, env_ids=None):
        """
        Compute observations for weighting network from multi-object data.

        IMPORTANT: Must return tensor of shape (len(env_ids), numWeightingObs)
        where numWeightingObs is defined in config (default: 30)

        Extracts per-object features from hoi_data multi-object section:
        - For each object: contact_obj (1) + mean_ig_distance (1) + contact_human_count (1) = 3 dims
        - For max_objects: 3 * max_objects dims
        - Default max_objects=3 → 9 dims, padded to 30 with zeros

        Args:
            env_ids: Environment indices to compute observations for

        Returns:
            weighting_obs: (len(env_ids), numWeightingObs) tensor
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Extract current reference observation for these envs
        curr_obs = self._curr_ref_obs[env_ids]  # (num_envs, total_obs_size)

        weighting_features = []

        for obj_idx in range(self.max_objects):
            # Extract components for this object from multi-object data section
            indices = self.multi_obj_component_index[obj_idx]

            # Extract features
            contact_obj = curr_obs[:, indices['contact_obj'][0]:indices['contact_obj'][1]]  # (num_envs, 1)
            ig = curr_obs[:, indices['ig'][0]:indices['ig'][1]]  # (num_envs, 156)
            contact_human = curr_obs[:, indices['contact_human'][0]:indices['contact_human'][1]]  # (num_envs, 52)

            # Summarize per object:
            # 1. Object contact flag (1 dim)
            # 2. Mean interaction graph distance (1 dim) - average distance from all body parts
            # 3. Number of body parts in contact (1 dim)
            ig_distances = ig.view(len(env_ids), 52, 3).norm(dim=-1)  # (num_envs, 52)
            ig_mean = ig_distances.mean(dim=-1, keepdim=True)  # (num_envs, 1)
            contact_count = (contact_human > 0.1).float().sum(dim=-1, keepdim=True)  # (num_envs, 1)

            obj_features = torch.cat([contact_obj, ig_mean, contact_count], dim=-1)  # (num_envs, 3)
            weighting_features.append(obj_features)

        # Concatenate all object features
        weighting_obs = torch.cat(weighting_features, dim=-1)  # (num_envs, max_objects * 3)

        # Verify size matches config (pad or truncate to match)
        expected_size = self._get_weighting_obs_size()
        actual_size = weighting_obs.shape[-1]
        if actual_size < expected_size:
            # Pad with zeros
            padding = torch.zeros((len(env_ids), expected_size - actual_size), device=self.device)
            weighting_obs = torch.cat([weighting_obs, padding], dim=-1)
        elif actual_size > expected_size:
            # Truncate
            weighting_obs = weighting_obs[:, :expected_size]

        return weighting_obs

# ========== Functions for observation modification ===============
 
    def _compute_observations_new(self, env_ids=None):
        current_obs = True
        ref_obs = True
        time_1 = time.time()
        body_pos = self._rigid_body_pos[env_ids]
        body_rot = self._rigid_body_rot[env_ids]
        body_vel = self._rigid_body_vel[env_ids]
        body_ang_vel = self._rigid_body_ang_vel[env_ids]

        contact_forces = self._contact_forces[env_ids]

        root_states = self._humanoid_root_states[env_ids]
        tar_states = self._target_states[env_ids]
        reward_tar_states = tar_states[:,0]
        time_2 = time.time()
        if current_obs: # 2048 980
            # ===== Current Observation Components =====
            human_obs = self._compute_proprioceptive_obs(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs) # (num_envs, 778)
            human_contact = self._compute_contact_obs(contact_forces) # (num_envs, 31)
            obj_obs = self.compute_obj_observations_simple(root_states, reward_tar_states) # (num_envs, 15)
            ig_obs = self._compute_ig_obs_simple(env_ids) # (num_envs, 156)
            time_3 = time.time()
        if ref_obs:  # 2048 628
            ts = self.progress_buf[env_ids].clone()
            delta_ts=[1,8]
            delta_ts_1=delta_ts[0]
            delta_ts_2=delta_ts[1]

            next_ts_1 = torch.clamp(ts + delta_ts_1, max=self.max_episode_length[self.data_id[env_ids]]-1)
            ref_obs_t1 = self.hoi_data[self.data_id[env_ids], next_ts_1].clone()

            next_ts_2 = torch.clamp(ts + delta_ts_2, max=self.max_episode_length[self.data_id[env_ids]]-1)
            ref_obs_t2 = self.hoi_data[self.data_id[env_ids], next_ts_2].clone()

            ref_human_obs_t1 = self._compute_reference_humanoid_obs(body_pos, body_rot, body_vel, body_ang_vel, ref_obs_t1, self._key_body_ids, contact_forces) # (num_envs, 550)
            ref_human_obs_t2 = self._compute_reference_humanoid_obs(body_pos, body_rot, body_vel, body_ang_vel, ref_obs_t2, self._key_body_ids, contact_forces) # (num_envs, 550)

            # Object reference observations
            ref_obj_obs_t1 = self._compute_reference_obj_obs(root_states, reward_tar_states, ref_obs_t1)  # (num_envs, 15)
            ref_obj_obs_t2 = self._compute_reference_obj_obs(root_states, reward_tar_states, ref_obs_t2)  # (num_envs, 15)

            # Interaction Graph reference observations
            ref_ig_obs_t1 = self._compute_reference_ig_obs(env_ids, ref_obs_t1)  # (num_envs, 63)
            ref_ig_obs_t2 = self._compute_reference_ig_obs(env_ids, ref_obs_t2)  # (num_envs, 63)
            time_4 = time.time()
        obs_current = torch.cat([human_obs, human_contact, obj_obs, ig_obs], dim=-1)
        t1_obs = torch.cat([ref_human_obs_t1, ref_obj_obs_t1, ref_ig_obs_t1], dim=-1)
        t2_obs = torch.cat([ref_human_obs_t2, ref_obj_obs_t2, ref_ig_obs_t2], dim=-1)
        obs = torch.cat([obs_current, t1_obs, t2_obs], dim=-1)
        self.obs_buf[env_ids] = obs
        time_5 = time.time()
        return obs

    def _compute_proprioceptive_obs(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs):
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        # heading rotation(rotation around z-axis), (2048, 4)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        root_h_obs = root_h

        # ------------- heading rot for body pos -------------
        heading_rot_expand = heading_rot.unsqueeze(-2) # (2048, 1, 4)
        heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1)) # heading_rot_expand: (2048, 52, 4)
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], # flat_heading_rot: (2048 * 52, 4)
                                                heading_rot_expand.shape[2])

        # ============= local_body_pos_obs =================
        root_pos_expand = root_pos.unsqueeze(-2) # (2048, 1, 3)
        local_body_pos = body_pos - root_pos_expand # (2048, 52, 3)
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]) # (2048 * 52, 3)
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos) # (2048 * 52, 3)
        local_body_pos_obs = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]) # (2048, 156)
        local_body_pos_obs = local_body_pos_obs[..., 3:] # (2048, 153) remove root pos 

        # ============= local_body_rot_obs =================
        flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]) # (2048 * 52, 4)
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot) # (2048 * 52, 4)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot) # (2048 * 52, 6)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]) # (2048, 312)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        # ============= local_body_vel =================
        flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]) # (2048, 156) to local coordinates of human heading direction
        
        # ============= local_body_ang_vel =================
        flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]) #(2048, 156) to local coordinates of human heading direction

        obs = torch.cat((root_h_obs, local_body_pos_obs, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
        return obs
    
    def compute_obj_observations_simple(self, root_states, reward_tar_states):
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        tar_pos = reward_tar_states[:, 0:3]
        tar_rot = reward_tar_states[:, 3:7]
        tar_vel = reward_tar_states[:, 7:10]
        tar_ang_vel = reward_tar_states[:, 10:13]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

        local_tar_pos = tar_pos - root_pos
        local_tar_pos[..., -1] = tar_pos[..., -1]
        local_tar_pos_obs = quat_rotate(heading_rot, local_tar_pos)
        local_tar_vel = quat_rotate(heading_rot, tar_vel)
        local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

        local_tar_rot = quat_mul(heading_rot, tar_rot)
        local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

        obs = torch.cat([local_tar_pos_obs, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)
        return obs

    def _compute_ig_obs_simple(self, env_ids):
        ig = self.extract_data_component('ig', obs=self._curr_obs[env_ids]).view(env_ids.shape[0], -1, 3) # (2048, 52, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True) # (2048, 52, 1)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp() # (2048, 52, 3) = (2048, 52, 3)*(2048, 52, 1)
        ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1) # (2048, 21*3)
        ig_all = ig_all.view(env_ids.shape[0], -1)    

        return ig_all # (2048, 156)

    def _compute_reference_humanoid_obs(self, body_pos, body_rot, body_vel, body_ang_vel, ref_obs, key_body_ids, contact_forces):
        """
        Compute reference-based observations including differences and reference states.

        Returns concatenated tensor of:
        - diff_local_body_pos_flat: Position difference between reference and current (key bodies only)
        - diff_local_body_rot_obs: Rotation difference between reference and current (no hand)
        - diff_body_contact: Contact difference between reference and current
        - local_ref_body_pos: Reference body positions in local frame (key bodies only)
        - local_ref_body_rot: Reference body rotations in local frame (no hand)
        - diff_local_vel: Velocity difference between reference and current (key bodies only)
        - diff_local_ang_vel: Angular velocity difference between reference and current (no hand)
        """
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        # Compute heading rotations
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        len_keypos = len(key_body_ids)

        # ----------- heading rot for key pos 21-----------
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1],
                                                heading_rot_expand_2.shape[2])

        # ----------- heading rot for 52body pos - 2 hands -----------
        heading_rot_expand_no_hand = heading_rot.unsqueeze(-2).repeat((1, 22, 1))
        flat_heading_rot_no_hand = heading_rot_expand_no_hand.reshape(heading_rot_expand_no_hand.shape[0] * heading_rot_expand_no_hand.shape[1],
                                                heading_rot_expand_no_hand.shape[2])
        # ----------- inv heading rot for 52body pos - 2 hands -----------
        heading_inv_rot_expand_no_hand = heading_inv_rot.unsqueeze(-2).repeat((1, 22, 1))
        flat_heading_inv_rot_no_hand = heading_inv_rot_expand_no_hand.reshape(heading_inv_rot_expand_no_hand.shape[0] * heading_inv_rot_expand_no_hand.shape[1],
                                                heading_inv_rot_expand_no_hand.shape[2])

        # ============= local body pos diff (2048, 63)=================
        _ref_body_pos = self.extract_data_component('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :] # (2048, 21, 3)
        _body_pos = body_pos[:, key_body_ids, :] # (2048, 21, 3)
        diff_global_body_pos = _ref_body_pos - _body_pos # (2048, 21, 3)
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3) # (2048, 63)

        # ============= Local reference body position (2048, 63)============= # 여기 ref가 아니라 state의 local pos같은데? intermimic에서도 이렇게 씀
        local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

        # ============= Local body rot diff (2048, 132)=================
        ref_body_rot = self.extract_data_component('body_rot', obs=ref_obs) # (2048, 208)
        ref_body_rot_no_hand = torch.cat((ref_body_rot[:, :18*4], ref_body_rot[:, 33*4:37*4]), dim=-1)
        body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot_no_hand)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        # ============= Local reference body-2hands rotation (2048, 132)=================
        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        # ============= local key_body vel diff (2048, 63)=================
        ref_body_vel = self.extract_data_component('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        _body_vel = body_vel[:, key_body_ids, :]
        diff_global_vel = ref_body_vel - _body_vel
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

        # ============= local body-2hands ang vel diff (2048, 66)=================
        ref_body_ang_vel = self.extract_data_component('body_rot_vel', obs=ref_obs)
        ref_body_ang_vel_no_hand = torch.cat((ref_body_ang_vel[:, :18*3], ref_body_ang_vel[:, 33*3:37*3]), dim=-1)
        body_ang_vel_no_hand = torch.cat((body_ang_vel[:, :18], body_ang_vel[:, 33:37]), dim=1)
        diff_global_ang_vel = ref_body_ang_vel_no_hand.view(-1, 22, 3) - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot_no_hand, diff_global_ang_vel.view(-1, 3)).view(-1, 22 * 3)

        # ============= 21body + 10finger_tips contact diff (2048, 31)=================
        ref_body_contact = self.extract_data_component('contact_human', obs=ref_obs)[:, self._contact_body_ids]
        body_contact_buf = contact_forces[:, self._contact_body_ids, :].clone()
        contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
        diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact) # 여기

        return torch.cat((diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact,
                         local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)

    def _compute_reference_obj_obs(self, root_states, reward_tar_states, ref_obs):
        """
        Compute reference-based object observations including differences and reference states.

        Returns concatenated tensor of:
        - diff_local_obj_pos_flat: Position difference between reference and current object
        - diff_local_obj_rot_obs: Rotation difference between reference and current object
        - diff_local_vel: Velocity difference between reference and current object
        - diff_local_ang_vel: Angular velocity difference between reference and current object
        """
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        tar_pos = reward_tar_states[:, 0:3]
        tar_rot = reward_tar_states[:, 3:7]
        tar_vel = reward_tar_states[:, 7:10]
        tar_ang_vel = reward_tar_states[:, 10:13]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        # --- Object position difference ---
        _ref_obj_pos = self.extract_data_component('obj_pos', obs=ref_obs)
        diff_global_obj_pos = _ref_obj_pos - tar_pos
        diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

        # --- Object rotation difference ---
        ref_obj_rot = self.extract_data_component('obj_rot', obs=ref_obs)
        diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
        diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)
        diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

        # --- Object velocity difference ---
        ref_obj_vel = self.extract_data_component('obj_pos_vel', obs=ref_obs)
        diff_global_vel = ref_obj_vel - tar_vel
        diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

        # --- Object angular velocity difference ---
        ref_obj_ang_vel = self.extract_data_component('obj_rot_vel', obs=ref_obs)
        diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
        diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)

        return torch.cat([diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)

    def _compute_reference_ig_obs(self, env_ids, ref_obs):
        """
        Compute reference-based interaction graph observations.

        Returns:
        - ref_ig - ig: Difference between reference and current interaction graph (key bodies only)
        """
        # Current interaction graph
        ig = self.extract_data_component('ig', obs=self._curr_obs[env_ids]).view(env_ids.shape[0], -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)

        # Reference interaction graph
        ref_ig = self.extract_data_component('ig', obs=ref_obs)
        ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)[:, self._key_body_ids, :]
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()
        ref_ig = ref_ig.view(env_ids.shape[0], -1)

        # Return difference between reference and current
        return ref_ig - ig

    def _compute_contact_obs(self, contact_forces):
        contact_body_ids = self._contact_body_ids
        body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
        contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()

        return contact # (2048, 31) 21 body + 10 finger tips
        

@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat