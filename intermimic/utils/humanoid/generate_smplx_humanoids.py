#!/usr/bin/env python3
"""
SMPL-X joint position mapper for humanoid XML generation
Maps SMPL-X body proportions to MuJoCo humanoid model'
Follows the rule from omomo.xml
"""

import pickle
import torch
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
import argparse

class GenerateXML:
    def __init__(self,
                smplx_model_path: str = "SMPL/models",
                enable_coord_transform: bool = False,):
        self.smplx_model_path = smplx_model_path
        self.smplx_model = None
        self.enable_coord_transform = enable_coord_transform
        self.finger_idx2_dict = {}
        self.hand_pose_ranges = {}
        
        # FromTo percentage rules
        self.fromto_rules = {
            'limb_joints': {
                'joints': ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle',
                          'L_Thorax', 'R_Thorax', 'L_Shoulder', 'R_Shoulder', 
                          'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'],
                'start_percent': 0.2,
                'end_percent': 0.8
            },
            'hand_joints': {
                'joints': ['L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3',
                          'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3',
                          'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
                          'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3',
                          'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3',
                          'R_Thumb1', 'R_Thumb2', 'R_Thumb3'],
                'start_percent': 0.2,
                'end_percent': 0.8
            },
            'spine_joints': {
                'joints': ['Torso', 'Spine', 'Neck'],
                'start_percent': 0.45,
                'end_percent': 0.55
            }
        }
        
        self.multi_children_joints = {
            'Chest': ['Neck', 'L_Thorax', 'R_Thorax'],
            'L_Wrist': ['L_Index1', 'L_Middle1', 'L_Pinky1', 'L_Ring1', 'L_Thumb1'],
            'R_Wrist': ['R_Index1', 'R_Middle1', 'R_Pinky1', 'R_Ring1', 'R_Thumb1']
        }
        
        self.original_fromto_values = {}
        
        # SMPL-X joint indices (55 joints)
        self.smplx_indices = {
            'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3,
            'left_knee': 4, 'right_knee': 5, 'spine2': 6, 'left_ankle': 7,
            'right_ankle': 8, 'spine3': 9, 'left_foot': 10, 'right_foot': 11,
            'neck': 12, 'left_collar': 13, 'right_collar': 14, 'head': 15,
            'left_shoulder': 16, 'right_shoulder': 17, 'left_elbow': 18,
            'right_elbow': 19, 'left_wrist': 20, 'right_wrist': 21,
            'left_index1': 25, 'left_index2': 26, 'left_index3': 27,
            'left_middle1': 28, 'left_middle2': 29, 'left_middle3': 30,
            'left_pinky1': 31, 'left_pinky2': 32, 'left_pinky3': 33,
            'left_ring1': 34, 'left_ring2': 35, 'left_ring3': 36,
            'left_thumb1': 37, 'left_thumb2': 38, 'left_thumb3': 39,
            'right_index1': 40, 'right_index2': 41, 'right_index3': 42,
            'right_middle1': 43, 'right_middle2': 44, 'right_middle3': 45,
            'right_pinky1': 46, 'right_pinky2': 47, 'right_pinky3': 48,
            'right_ring1': 49, 'right_ring2': 50, 'right_ring3': 51,
            'right_thumb1': 52, 'right_thumb2': 53, 'right_thumb3': 54
        }
        # SMPL-X to omomo.xml joint mapping
        self.joint_mapping = {
            'pelvis': 'Pelvis',
            'left_hip': 'L_Hip', 'right_hip': 'R_Hip',
            'left_knee': 'L_Knee', 'right_knee': 'R_Knee', 
            'left_ankle': 'L_Ankle', 'right_ankle': 'R_Ankle',
            'left_foot': 'L_Toe', 'right_foot': 'R_Toe',
            'spine1': 'Torso', 'spine2': 'Spine', 'spine3': 'Chest',
            'neck': 'Neck', 'head': 'Head',
            'left_collar': 'L_Thorax', 'right_collar': 'R_Thorax',
            'left_shoulder': 'L_Shoulder', 'right_shoulder': 'R_Shoulder',
            'left_elbow': 'L_Elbow', 'right_elbow': 'R_Elbow',
            'left_wrist': 'L_Wrist', 'right_wrist': 'R_Wrist',
            'left_index1': 'L_Index1', 'left_index2': 'L_Index2', 'left_index3': 'L_Index3',
            'left_middle1': 'L_Middle1', 'left_middle2': 'L_Middle2', 'left_middle3': 'L_Middle3',
            'left_pinky1': 'L_Pinky1', 'left_pinky2': 'L_Pinky2', 'left_pinky3': 'L_Pinky3',
            'left_ring1': 'L_Ring1', 'left_ring2': 'L_Ring2', 'left_ring3': 'L_Ring3',
            'left_thumb1': 'L_Thumb1', 'left_thumb2': 'L_Thumb2', 'left_thumb3': 'L_Thumb3',
            'right_index1': 'R_Index1', 'right_index2': 'R_Index2', 'right_index3': 'R_Index3',
            'right_middle1': 'R_Middle1', 'right_middle2': 'R_Middle2', 'right_middle3': 'R_Middle3',
            'right_pinky1': 'R_Pinky1', 'right_pinky2': 'R_Pinky2', 'right_pinky3': 'R_Pinky3',
            'right_ring1': 'R_Ring1', 'right_ring2': 'R_Ring2', 'right_ring3': 'R_Ring3',
            'right_thumb1': 'R_Thumb1', 'right_thumb2': 'R_Thumb2', 'right_thumb3': 'R_Thumb3',
        }
    
    def load_smplx_model(self, gender: str):
        """Load SMPL-X model"""
        try:
            import smplx
            self.smplx_model = smplx.create(
                model_path=self.smplx_model_path,
                model_type="smplx",
                flat_hand_mean=True,
                use_pca=False,
                num_betas=20,
                num_expression_coeffs=10,
                gender=gender,
                ext='pkl'
            )
            return True
        except Exception as e:
            print(f"Error loading SMPL-X: {e}")
            return False
    
    def extract_smplx_joint_positions(self, beta: torch.Tensor, gender: str):
        """Extract SMPL-X joint positions"""
        if not self.load_smplx_model(gender):
            return None
        
        beta_tensor = beta.unsqueeze(0) if beta.dim() == 1 else beta
        output = self.smplx_model(betas=beta_tensor)
        joint_positions = output.joints[0].detach().numpy() # (127, 3)
        
        mapped_positions = {}
        for smplx_name, idx in self.smplx_indices.items():
            if smplx_name in self.joint_mapping:
                omomo_name = self.joint_mapping[smplx_name]
                mapped_positions[omomo_name] = joint_positions[idx]
        
        return mapped_positions

    def extract_hand_pose_ranges(self, scene_path: str, buffer_deg: float = 10.0):
        """Extract hand pose ranges from SMPL-X sequence data"""
        hand_pose_order = [
            'left_index1_x', 'left_index1_y', 'left_index1_z',
            'left_index2_x', 'left_index2_y', 'left_index2_z',
            'left_index3_x', 'left_index3_y', 'left_index3_z',
            'left_middle1_x', 'left_middle1_y', 'left_middle1_z',
            'left_middle2_x', 'left_middle2_y', 'left_middle2_z',
            'left_middle3_x', 'left_middle3_y', 'left_middle3_z',
            'left_pinky1_x', 'left_pinky1_y', 'left_pinky1_z',
            'left_pinky2_x', 'left_pinky2_y', 'left_pinky2_z',
            'left_pinky3_x', 'left_pinky3_y', 'left_pinky3_z',
            'left_ring1_x', 'left_ring1_y', 'left_ring1_z',
            'left_ring2_x', 'left_ring2_y', 'left_ring2_z',
            'left_ring3_x', 'left_ring3_y', 'left_ring3_z',
            'left_thumb1_x', 'left_thumb1_y', 'left_thumb1_z',
            'left_thumb2_x', 'left_thumb2_y', 'left_thumb2_z',
            'left_thumb3_x', 'left_thumb3_y', 'left_thumb3_z',
            'right_index1_x', 'right_index1_y', 'right_index1_z',
            'right_index2_x', 'right_index2_y', 'right_index2_z',
            'right_index3_x', 'right_index3_y', 'right_index3_z',
            'right_middle1_x', 'right_middle1_y', 'right_middle1_z',
            'right_middle2_x', 'right_middle2_y', 'right_middle2_z',
            'right_middle3_x', 'right_middle3_y', 'right_middle3_z',
            'right_pinky1_x', 'right_pinky1_y', 'right_pinky1_z',
            'right_pinky2_x', 'right_pinky2_y', 'right_pinky2_z',
            'right_pinky3_x', 'right_pinky3_y', 'right_pinky3_z',
            'right_ring1_x', 'right_ring1_y', 'right_ring1_z',
            'right_ring2_x', 'right_ring2_y', 'right_ring2_z',
            'right_ring3_x', 'right_ring3_y', 'right_ring3_z',
            'right_thumb1_x', 'right_thumb1_y', 'right_thumb1_z',
            'right_thumb2_x', 'right_thumb2_y', 'right_thumb2_z',
            'right_thumb3_x', 'right_thumb3_y', 'right_thumb3_z',
        ]

        # Mapping from hand pose order to XML joint names
        smplx_to_xml_mapping = {
            'left_index1': 'L_Index1', 'left_index2': 'L_Index2', 'left_index3': 'L_Index3',
            'left_middle1': 'L_Middle1', 'left_middle2': 'L_Middle2', 'left_middle3': 'L_Middle3',
            'left_pinky1': 'L_Pinky1', 'left_pinky2': 'L_Pinky2', 'left_pinky3': 'L_Pinky3',
            'left_ring1': 'L_Ring1', 'left_ring2': 'L_Ring2', 'left_ring3': 'L_Ring3',
            'left_thumb1': 'L_Thumb1', 'left_thumb2': 'L_Thumb2', 'left_thumb3': 'L_Thumb3',
            'right_index1': 'R_Index1', 'right_index2': 'R_Index2', 'right_index3': 'R_Index3',
            'right_middle1': 'R_Middle1', 'right_middle2': 'R_Middle2', 'right_middle3': 'R_Middle3',
            'right_pinky1': 'R_Pinky1', 'right_pinky2': 'R_Pinky2', 'right_pinky3': 'R_Pinky3',
            'right_ring1': 'R_Ring1', 'right_ring2': 'R_Ring2', 'right_ring3': 'R_Ring3',
            'right_thumb1': 'R_Thumb1', 'right_thumb2': 'R_Thumb2', 'right_thumb3': 'R_Thumb3',
        }

        try:
            scene_path = Path(scene_path)
            smplx_pose_path = scene_path / "smplx_pose.pkl"

            with open(smplx_pose_path, 'rb') as f:
                smplx_seq = pickle.load(f)

            hand_pose = smplx_seq['hand_pose']  # Shape: (num_frames, 90)

            # Extract ranges for each joint axis
            for i, joint_axis in enumerate(hand_pose_order):
                # Convert radians to degrees
                values_deg = hand_pose[:, i] * 180 / np.pi
                min_val = float(values_deg.min()) - buffer_deg
                max_val = float(values_deg.max()) + buffer_deg

                # Parse joint name and axis
                joint_base = '_'.join(joint_axis.split('_')[:-1])  # e.g., 'left_index1'
                axis = joint_axis.split('_')[-1]  # e.g., 'x'

                if joint_base in smplx_to_xml_mapping:
                    xml_joint_name = smplx_to_xml_mapping[joint_base]
                    joint_axis_name = f"{xml_joint_name}_{axis}"

                    self.hand_pose_ranges[joint_axis_name] = (min_val, max_val)

            print(f"Extracted hand pose ranges for {len(self.hand_pose_ranges)} joint axes")
            return True

        except Exception as e:
            print(f"Warning: Could not extract hand pose ranges: {e}")
            return False

    def _apply_coord_transform(self, pos_array: np.ndarray) -> np.ndarray:
        """Apply coordinate transformation if enabled"""
        if self.enable_coord_transform:
            return np.array([pos_array[2], pos_array[0], pos_array[1]])
        return pos_array.copy()
    
    def _transform_joint_axis(self, axis_str: str) -> str:
        """Transform joint axis based on coordinate transformation"""
        if not self.enable_coord_transform:
            return axis_str
        
        axis = [float(x) for x in axis_str.split()]
        if len(axis) != 3:
            return axis_str
        
        transformed = [axis[2], axis[0], axis[1]]
        return f"{transformed[0]:.0f} {transformed[1]:.0f} {transformed[2]:.0f}"
    

    def _load_original_fromto_values(self, omomo_xml_path: str):
        """Load original fromto values from omomo.xml"""
        tree = ET.parse(omomo_xml_path)
        root = tree.getroot()
        
        def parse_fromto_recursive(elem):
            for body in elem.findall('body'):
                body_name = body.get('name')
                if body_name:
                    for geom in body.findall('geom'):
                        fromto_str = geom.get('fromto')
                        if fromto_str:
                            self.original_fromto_values[body_name] = fromto_str
                            break
                    parse_fromto_recursive(body)
        
        worldbody = root.find('worldbody')
        if worldbody is not None:
            parse_fromto_recursive(worldbody)
    
    def update_xml_joint_positions(self, omomo_xml_path: str, joint_positions: Dict[str, np.ndarray], scene_name: str):
        """Update XML with SMPL-X joint positions"""

        tree = ET.parse(omomo_xml_path)
        root = tree.getroot()
        
        self._load_original_fromto_values(omomo_xml_path)
        root.set('model', f"smplx_mapped_{scene_name}")
        
        hierarchy = self._build_joint_hierarchy_from_xml(root)
        
        def update_body_recursive(parent_elem, parent_body_name=None):
            for body in parent_elem.findall('body'):
                body_name = body.get('name')
                if body_name and body_name in joint_positions:
                    if body_name == 'Pelvis':
                        new_pos = joint_positions[body_name]
                    elif parent_body_name and parent_body_name in joint_positions:
                        new_pos = joint_positions[body_name] - joint_positions[parent_body_name]
                    else:
                        new_pos = joint_positions[body_name]
                    
                    swapped_pos = self._apply_coord_transform(new_pos)
                    body.set('pos', f"{swapped_pos[0]:.4f} {swapped_pos[1]:.4f} {swapped_pos[2]:.4f}")
                    
                    self._update_fromto_in_body(body, body_name, joint_positions, hierarchy)
                    self._update_joint_axes_in_body(body)
                    self._set_adaptive_finger_ranges(body, body_name)
                    self._set_non_hand_joint_ranges(body, body_name)
                    self._scale_finger_sizes(body, body_name)
                    
                    update_body_recursive(body, body_name)
                else:
                    # Still apply finger modifications even for bodies without SMPL-X positions
                    self._set_adaptive_finger_ranges(body, body_name)
                    self._set_non_hand_joint_ranges(body, body_name)
                    self._scale_finger_sizes(body, body_name)
                    update_body_recursive(body, body_name)
        
        worldbody = root.find('worldbody')
        if worldbody is not None:
            update_body_recursive(worldbody)
        
        return root
    
    def _build_joint_hierarchy_from_xml(self, root):
        """Build joint hierarchy from XML"""
        hierarchy = {}
        
        def parse_hierarchy(parent_elem, parent_name=None):
            for body in parent_elem.findall('body'):
                body_name = body.get('name')
                if body_name and parent_name:
                    hierarchy[body_name] = parent_name
                parse_hierarchy(body, body_name)
        
        worldbody = root.find('worldbody')
        if worldbody is not None:
            parse_hierarchy(worldbody)
        
        return hierarchy
    
    def _get_fromto_rule(self, body_name: str):
        """Get FromTo percentage rule for a joint"""
        for rule_data in self.fromto_rules.values():
            if body_name in rule_data['joints']:
                return rule_data
        return self.fromto_rules['limb_joints']
    
    def _get_child_joints(self, body_name: str, hierarchy: Dict[str, str]):
        """Get child joints for a body"""
        return [child for child, parent in hierarchy.items() if parent == body_name]
    
    def _calculate_children_average(self, body_name: str, joint_positions: Dict[str, np.ndarray], hierarchy: Dict[str, str]):
        """Calculate average position of children joints"""
        children = self.multi_children_joints.get(body_name, self._get_child_joints(body_name, hierarchy))
        
        if not children:
            return np.array([0.0, 0.0, 0.0])
        
        parent_pos = joint_positions.get(body_name, np.array([0.0, 0.0, 0.0]))
        child_positions = []
        
        for child in children:
            if child in joint_positions:
                relative_pos = joint_positions[child] - parent_pos
                child_positions.append(relative_pos)
        
        return np.mean(child_positions, axis=0) if child_positions else np.array([0.0, 0.0, 0.0])
    
    def _update_fromto_in_body(self, body_elem, body_name: str, joint_positions: Dict[str, np.ndarray], hierarchy: Dict[str, str]):
        """Update fromto attributes"""
        # Set specific sizes for certain joints
        size_mapping = {
            'Pelvis': '0.09',
            'Torso': '0.08',
            'Spine': '0.06',
            'Chest': '0.08'
        }
        
        for geom in body_elem.findall('geom'):
            # Update size if specified
            if body_name in size_mapping:
                geom.set('size', size_mapping[body_name])
            
            # Handle box type geoms - swap pos and size when NOT using transform
            if geom.get('type') == 'box' and not self.enable_coord_transform:
                # Swap pos if exists
                pos_attr = geom.get('pos')
                if pos_attr:
                    pos = [float(x) for x in pos_attr.split()]
                    if len(pos) == 3:
                        swapped_pos = [pos[1], pos[2], pos[0]]  # [a,b,c] -> [b,c,a]
                        geom.set('pos', f"{swapped_pos[0]:.4f} {swapped_pos[1]:.4f} {swapped_pos[2]:.4f}")
                
                # Swap size if exists  
                size_attr = geom.get('size')
                if size_attr:
                    size = [float(x) for x in size_attr.split()]
                    if len(size) == 3:
                        swapped_size = [size[1], size[2], size[0]]  # [a,b,c] -> [b,c,a]
                        geom.set('size', f"{swapped_size[0]:.4f} {swapped_size[1]:.4f} {swapped_size[2]:.4f}")
            import math
            if geom.get('fromto') is not None:
                rule = self._get_fromto_rule(body_name)
                
                if body_name in self.multi_children_joints or len(self._get_child_joints(body_name, hierarchy)) > 1:
                    target_pos = self._calculate_children_average(body_name, joint_positions, hierarchy)
                else:
                    children = self._get_child_joints(body_name, hierarchy)
                    if children and children[0] in joint_positions:
                        parent_pos = joint_positions.get(body_name, np.array([0.0, 0.0, 0.0]))
                        child_pos = joint_positions[children[0]]
                        target_pos = child_pos - parent_pos
                    else:
                        # Check if this is a finger end joint that needs coordinate transformation
                        finger_end_joints = ['L_Index3', 'L_Middle3', 'L_Pinky3', 'L_Ring3', 'L_Thumb3',
                                            'R_Index3', 'R_Middle3', 'R_Pinky3', 'R_Ring3', 'R_Thumb3']
                        
                        original_fromto = self.original_fromto_values.get(body_name)
                        if original_fromto:
                            # If this is a finger end joint and we're NOT using --transform flag,
                            # apply the coordinate transformation [a,b,c] -> [c,a,b] to the fromto
                            if body_name in finger_end_joints:
                                parent_fromto = self.finger_idx2_dict[body_name.replace('3', '2')]
                                fromto_values = [float(x) for x in original_fromto.split()]
                                child_size = math.sqrt((fromto_values[3] - fromto_values[0])**2 \
                                    + (fromto_values[4] - fromto_values[1])**2 \
                                    + (fromto_values[5] - fromto_values[2])**2)
                                parent_size = math.sqrt((parent_fromto[3] - parent_fromto[0])**2 \
                                    + (parent_fromto[4] - parent_fromto[1])**2 \
                                    + (parent_fromto[5] - parent_fromto[2])**2)
                                
                                start = np.array(parent_fromto[:3])
                                end = np.array(parent_fromto[3:])
                                
                                start_new = (start + end) / 2 - (child_size / 2) * (end - start) / parent_size
                                end_new = (start + end) / 2 + (child_size / 2) * (end - start) / parent_size

                                transformed_fromto = f"{start_new[0]:.4f} {start_new[1]:.4f} {start_new[2]:.4f} " \
                                                    f"{end_new[0]:.4f} {end_new[1]:.4f} {end_new[2]:.4f}"
                                geom.set('fromto', transformed_fromto)
                            else:
                                geom.set('fromto', original_fromto)
                            break
                        continue
                
                fromto_start = target_pos * rule['start_percent']
                fromto_end = target_pos * rule['end_percent']
                
                fromto_start_swapped = self._apply_coord_transform(fromto_start)
                fromto_end_swapped = self._apply_coord_transform(fromto_end)
                
                fromto = f"{fromto_start_swapped[0]:.4f} {fromto_start_swapped[1]:.4f} {fromto_start_swapped[2]:.4f} " \
                        f"{fromto_end_swapped[0]:.4f} {fromto_end_swapped[1]:.4f} {fromto_end_swapped[2]:.4f}"
                geom.set('fromto', fromto)
                if body_name[-1] == '2':
                    self.finger_idx2_dict[body_name] = [fromto_start_swapped[0], fromto_start_swapped[1], fromto_start_swapped[2], 
                                                        fromto_end_swapped[0], fromto_end_swapped[1], fromto_end_swapped[2]]

                break
    
    def _update_joint_axes_in_body(self, body_elem):
        """Update joint axes based on coordinate transformation or axis reordering"""
        for joint in body_elem.findall('joint'):
            axis_attr = joint.get('axis')
            if axis_attr is not None:
                # Apply coordinate transformation if enabled
                transformed_axis = self._transform_joint_axis(axis_attr)
                
                if transformed_axis != axis_attr:
                    joint.set('axis', transformed_axis)
    
    def _set_adaptive_finger_ranges(self, body_elem, body_name):
        """Set adaptive range values for finger joints based on SMPL-X sequence data"""
        # Define all finger joint base names (without _x, _y, _z suffixes)
        finger_joints = [
            'L_Index1', 'L_Index2', 'L_Index3',
            'L_Middle1', 'L_Middle2', 'L_Middle3',
            'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
            'L_Ring1', 'L_Ring2', 'L_Ring3',
            'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
            'R_Index1', 'R_Index2', 'R_Index3',
            'R_Middle1', 'R_Middle2', 'R_Middle3',
            'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
            'R_Ring1', 'R_Ring2', 'R_Ring3',
            'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
        ]

        # Only process if this is a finger joint
        if body_name not in finger_joints:
            return

        # Find all joints for this finger (x, y, z axes)
        finger_joints_dict = {}

        for joint in body_elem.findall('joint'):
            joint_name = joint.get('name')
            if joint_name:
                for axis in ['x', 'y', 'z']:
                    if joint_name == f"{body_name}_{axis}":
                        finger_joints_dict[axis] = joint

        # Set adaptive ranges based on SMPL-X data
        for axis, joint_elem in finger_joints_dict.items():
            range_key = f"{body_name}_{axis}"

            if range_key in self.hand_pose_ranges:
                min_val, max_val = self.hand_pose_ranges[range_key]
                new_range = f"{min_val:.2f} {max_val:.2f}"
                joint_elem.set('range', new_range)
                print(f"Set adaptive range for {range_key}: {new_range}")
            else:
                # Fallback: keep original range if no SMPL-X data available
                original_range = joint_elem.get('range')
                print(f"Warning: No SMPL-X data for {range_key}, keeping original range: {original_range}")

    def _set_non_hand_joint_ranges(self, body_elem, body_name):
        """Set range values for non-hand joints to -150 to 150 degrees"""
        # Define all non-hand joint base names
        non_hand_joints = [
            # Legs
            'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe',
            # Torso and spine
            'Torso', 'Spine', 'Chest', 'Neck', 'Head',
            # Arms
            'L_Thorax', 'R_Thorax', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
        ]

        # Only process if this is a non-hand joint
        if body_name not in non_hand_joints:
            return

        # Set range -150 to 150 for all joints in this body
        for joint in body_elem.findall('joint'):
            joint_name = joint.get('name')
            if joint_name:
                # Set range to -150 to 150 degrees
                joint.set('range', '-150.00 150.00')
                print(f"Set non-hand joint range for {joint_name}: -150.00 150.00")

    def _scale_finger_sizes(self, body_elem, body_name, scale_factor=0.7):
        """Scale finger geometry sizes to specified factor (default 70%)"""
        # Define all finger joint base names
        finger_joints = [
            'L_Index1', 'L_Index2', 'L_Index3', 
            'L_Middle1', 'L_Middle2', 'L_Middle3',
            'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 
            'L_Ring1', 'L_Ring2', 'L_Ring3',
            'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
            'R_Index1', 'R_Index2', 'R_Index3', 
            'R_Middle1', 'R_Middle2', 'R_Middle3',
            'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 
            'R_Ring1', 'R_Ring2', 'R_Ring3',
            'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
        ]
        
        # Only process if this is a finger joint
        if body_name not in finger_joints:
            return
        
        # Scale all geometry sizes in this finger body
        for geom in body_elem.findall('geom'):
            size_attr = geom.get('size')
            if size_attr is not None:
                try:
                    # Parse size values (can be single value or multiple values)
                    size_values = [float(x) for x in size_attr.split()]
                    
                    # Scale all size values
                    scaled_values = [val * scale_factor for val in size_values]
                    
                    # Format back to string
                    scaled_size = ' '.join(f"{val:.4f}" for val in scaled_values)
                    geom.set('size', scaled_size)
                    
                    print(f"Scaled size for {body_name}: {size_attr} -> {scaled_size}")
                    
                except ValueError:
                    print(f"Warning: Could not parse size '{size_attr}' for {body_name}")
                    continue
    
    
    def generate_xml(self, scene_path: str, output_dir: str = "intermimic/data/assets/parahome/sim_human"):
        """Generate personalized humanoid with SMPL-X joint positions"""
        scene_name = Path(scene_path).name
        print(f"Generating SMPL-X humanoid: {scene_name}")

        scene_path = Path(scene_path)
        shape_param_path = scene_path / "smplx_params.pkl"
        shape_param = pickle.load(open(shape_param_path, "rb"))
        beta = shape_param['beta'][0]
        gender = shape_param['gender']

        # Extract hand pose ranges from SMPL-X sequence data
        self.extract_hand_pose_ranges(str(scene_path))

        joint_positions = self.extract_smplx_joint_positions(beta, gender)

        
        omomo_xml_path = "intermimic/data/assets/smplx/omomo.xml"
        xml_root = self.update_xml_joint_positions(omomo_xml_path, joint_positions, scene_name)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        xml_file = output_path / f"{scene_name}.xml"
        
        self._indent_xml(xml_root)
        tree = ET.ElementTree(xml_root)
        tree.write(xml_file, encoding='utf-8', xml_declaration=True)
        
        print(f"Generated: {xml_file}")
        return True
    
    def _indent_xml(self, elem, level=0):
        """Add indentation to XML"""
        indent = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

def main():
    """Generate SMPL-X joint mapped humanoids"""
    parser = argparse.ArgumentParser(description="Generate SMPL-X joint mapped humanoids")
    parser.add_argument("--transform", action="store_true", help="Enable coordinate transformation, generate z-up humanoid")
    parser.add_argument("--scenes", nargs="+", type=int, default=[110], help="List of scene numbers to process")
    parser.add_argument("--output_dir", type=str, default="intermimic/data/assets/parahome/sim_human", help="Output directory")
    args = parser.parse_args()
    
    parahome_seq_path = Path("intermimic/data/assets/parahome/smplx_seq")
    if args.scenes == [-1]:
        args.scenes = list(range(1, 208, 1))

    scenes_name = [f"s{scene}" for scene in args.scenes]
    scenes = [parahome_seq_path / scene_name for scene_name in scenes_name]
    
    generator = GenerateXML(enable_coord_transform=args.transform)
    success_count = 0

    for scene_path in scenes:
        if scene_path.is_dir():
            try:
                if generator.generate_xml(str(scene_path), args.output_dir):
                    success_count += 1
            except Exception as e:
                print(f"Error: {e}")
    
    print(f"\nGenerated {success_count}/{len(scenes)} humanoids")

if __name__ == "__main__":
    main()