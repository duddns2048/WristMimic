#!/usr/bin/env python3
"""
Compact humanoid controller using utils.py functions
"""

import os
import numpy as np
from pathlib import Path
from isaacgym import gymapi, gymutil
import torch
from sim_utils import *

class HumanoidController:
    def __init__(self, xml_path=None):
        self.xml_path = Path(xml_path)
        self.joint_to_dof_indices = {}
        
        # Simplified joint angles - only non-zero ones
        self.joint_angles = {
            # 'l_hip': np.array([0, 1, 2.5]),
            # # 'l_elbow': np.array([0, 2.2, 0]),
            # 'l_index1': np.array([0, 0, -1.57])
        }
        
        # Root pose
        self.root_position = np.array([0.0, 0.0, 0.95])
        self.root_orientation = np.array([0.0, 0.0, 0.0])  # axis-angle

        
    def initialize(self, args):
        """Initialize Isaac Gym and load humanoid"""
        self.gym = gymapi.acquire_gym()
        
        # Setup simulation with standard parameters
        self.sim = self._create_simulation(args)
        if not self.sim:
            return False
        
        # Load humanoid asset
        asset = self._load_humanoid_asset()
        if not asset:
            return False
        
        # Get asset info and build joint mapping
        self.num_dofs = self.gym.get_asset_dof_count(asset)
        self.dof_names = self.gym.get_asset_dof_names(asset)
        self._build_joint_mapping()

        print(f"DOF names: {self.dof_names}")
        print(f"Joint mapping: {self.joint_to_dof_indices}")
        print(f"Body joint names: {get_body_joint_names()}")
        
        print(f"✓ Humanoid loaded: {self.num_dofs} DOFs, {self.gym.get_asset_rigid_body_count(asset)} bodies")
        
        # Setup environment and actor
        self._setup_environment()
        if not hasattr(self, 'viewer') or not self.viewer:
            return False
            
        self.actor = self._create_humanoid_actor(asset)
        if self.actor is None:
            return False
            
        self._configure_dof_properties()
        self._setup_camera()
        
        return True
    
    def _create_simulation(self, args):
        """Create simulation with standard parameters"""
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1.0 / 60.0
        sim_params.physx.solver_type = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.use_gpu_pipeline = False
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        # sim_params.physx.contact_offset = 0.01
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity: 0.2
        sim_params.physx.default_buffer_size_multiplier: 10.0
        
        sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if sim:
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            self.gym.add_ground(sim, plane_params)
        return sim
    
    def _load_humanoid_asset(self):
        """Load humanoid asset with appropriate options"""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = True
        asset_options.override_inertia = True
        
        return self.gym.load_asset(self.sim, str(self.xml_path.parent), self.xml_path.name, asset_options)
    
    def _build_joint_mapping(self):
        """Build mapping from joint names to DOF indices"""
        for i, dof_name in enumerate(self.dof_names):
            parts = dof_name.split('_')
            if len(parts) >= 2:
                axis = parts[-1]
                joint_name = '_'.join(parts[:-1])
                if joint_name not in self.joint_to_dof_indices:
                    self.joint_to_dof_indices[joint_name] = {}
                self.joint_to_dof_indices[joint_name][axis] = i
    
    def _setup_environment(self):
        """Setup viewer and environment"""
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        spacing = 10.0
        self.env = self.gym.create_env(self.sim, 
                                     gymapi.Vec3(-spacing, -spacing, 0.0),
                                     gymapi.Vec3(spacing, spacing, spacing), 1)
    
    def _create_humanoid_actor(self, asset):
        """Create humanoid actor"""
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        return self.gym.create_actor(self.env, asset, pose, "humanoid", 0, 0)
    
    def _configure_dof_properties(self):
        """Configure DOF properties for position control"""
        dof_props = self.gym.get_actor_dof_properties(self.env, self.actor)
        dof_props['stiffness'][:] = 2000.0
        dof_props['damping'][:] = 200.0
        dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
        self.gym.set_actor_dof_properties(self.env, self.actor, dof_props)
    
    def _setup_camera(self):
        """Setup camera position"""
        self.gym.viewer_camera_look_at(self.viewer, None, 
                                     gymapi.Vec3(3.0, 3.0, 5.0), 
                                     gymapi.Vec3(0.0, 0.0, 1.0))
    
    def update_robot_pose(self):
        """Apply current joint angles using utils functions"""
        dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
        
        # Use body joint names from utils and apply only configured angles
        body_joint_names = get_body_joint_names()
        angle_keys = ['l_hip', 'r_hip', 'torso', 'l_knee', 'r_knee', 'spine',
                     'l_ankle', 'r_ankle', 'chest', 'l_toe', 'r_toe', 'neck',
                     'l_thorax', 'r_thorax', 'head', 'l_shoulder', 'r_shoulder',
                     'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
        
        # Apply joint angles using utils functions
        for isaac_joint, angle_key in zip(body_joint_names, angle_keys):
            if isaac_joint in self.joint_to_dof_indices and angle_key in self.joint_angles:
                joint_axis_angle = self.joint_angles[angle_key]
                dof_indices = self.joint_to_dof_indices[isaac_joint]
                
                # Use utils functions for axis-angle conversion
                if np.linalg.norm(joint_axis_angle) < 1e-5:
                    rx, ry, rz = 0.0, 0.0, 0.0
                else:
                    rx, ry, rz = joint_axis_angle[0], joint_axis_angle[1], joint_axis_angle[2]
                
                if 'x' in dof_indices:
                    dof_states['pos'][dof_indices['x']] = rx
                if 'y' in dof_indices:
                    dof_states['pos'][dof_indices['y']] = ry
                if 'z' in dof_indices:
                    dof_states['pos'][dof_indices['z']] = rz
        # dof_states['pos'][36:39] = np.array([1.57, 0, -1.57])
        # dof_states['pos'][96:99] = np.array([0, 0, -1.57])
        self.gym.set_actor_dof_states(self.env, self.actor, dof_states, gymapi.STATE_ALL)

    def update_root_state(self):
        """Update root position and orientation using utils function"""
        # Convert axis-angle to quaternion
        angle = np.linalg.norm(self.root_orientation)
        if angle < 1e-6:
            rotation = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            axis = self.root_orientation / angle
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)
            rotation = gymapi.Quat(
                float(axis[0] * sin_half), float(axis[1] * sin_half),
                float(axis[2] * sin_half), float(cos_half)
            )
        
        position = gymapi.Vec3(float(self.root_position[0]), 
                              float(self.root_position[1]), 
                              float(self.root_position[2]))
        
        # Use utils function for updating actor transform
        update_actor_transform_via_sim_state(self.gym, self.sim, self.env, self.actor, position, rotation)
    
    def set_joint_angles_from_dict(self, angles_dict):
        """Update joint angles from dictionary"""
        for joint_name, angles in angles_dict.items():
            self.joint_angles[joint_name] = np.array(angles)
    
    def run(self):
        """Main simulation loop"""
        print("\n=== Humanoid Controller Running ===")
        print(f"Active joint angles: {list(self.joint_angles.keys())}")
        print("Press ESC to exit\n")
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.update_root_state()
            self.update_robot_pose()
            
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'viewer') and self.viewer:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, 'sim') and self.sim:
            self.gym.destroy_sim(self.sim)
        print("Cleanup completed")

def main():
    """Main function"""
    custom_parameters = [{"name": "--path", "type": str, "help": "humanoid path"}]
    args = gymutil.parse_arguments(description="Humanoid Controller", custom_parameters=custom_parameters)
    
    controller = HumanoidController(args.path)
    
    # Initialize the controller
    if not controller.initialize(args):
        print("Failed to initialize controller")
        return 1
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main())