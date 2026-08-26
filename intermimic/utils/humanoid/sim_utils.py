import json
import os
import pickle
from isaacgym import gymapi
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET


def load_scene_objects(scene_number):
    with open(f"data/ParaHome/data/seq/s{scene_number}/object_in_scene.json", 'r') as f:
        scene_objects = json.load(f)
    enabled_count = sum(1 for enabled in scene_objects.values() if enabled)
    print(f"✓ Loaded scene-specific objects: {enabled_count} enabled")
    return scene_objects


def load_human_sequence_data(scene_number):
    seq_dir = f"data/ParaHome/data/smplx_seq/s{scene_number}"
    seq_file = os.path.join(seq_dir, os.listdir(seq_dir)[1])
    
    with open(seq_file, 'rb') as f:
        smplx_seq = pickle.load(f)
    
    human_frames = smplx_seq['body_pose'].shape[0]
    print(f"✓ Loaded human sequence: {human_frames} frames")
    
    return smplx_seq['body_pose'], smplx_seq['hand_pose'], smplx_seq['global_orient'], smplx_seq['transl'], human_frames


def load_object_sequence_data(scene_number):
    seq_path = f"data/ParaHome/data/seq/s{scene_number}"
    
    with open(os.path.join(seq_path, "object_transformations.pkl"), 'rb') as f:
        rb_seq = pickle.load(f)
    with open(os.path.join(seq_path, "joint_states.pkl"), 'rb') as f:
        joint_seq = pickle.load(f)
    
    print(f"✓ Loaded object sequence: {len(rb_seq)} frames")
    return rb_seq, joint_seq, len(rb_seq)


def load_contact_data(scene_number):
    with open(f"data/ParaHome/data/sim_obj_contact/s{scene_number}.pkl", 'rb') as f:
        return pickle.load(f)


def load_pelvis_offset_from_xml(scene_number, xml_path):
    tree = ET.parse(xml_path)
    pelvis_body = tree.getroot().find(".//body[@name='Pelvis']")
    pos_values = [float(x) for x in pelvis_body.get('pos').split()]
    print(f"✓ Loaded pelvis offset from XML (scene {scene_number}): {pos_values}")
    return pos_values


def matrix_to_gym_transform(H):
    translation = H[:3, 3]
    position = gymapi.Vec3(float(translation[0]), float(translation[1]), float(translation[2]))
    
    quat = R.from_matrix(H[:3, :3]).as_quat()
    rotation = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
    
    return position, rotation


def update_actor_transform_via_sim_state(gym, sim, env, actor, position, rotation):
    """Update actor transform using sim-level rigid body states (no warnings)"""
    rb_states = gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL)
    
    num_actors = gym.get_actor_count(env)
    rb_index = 0
    
    # Calculate global rigid body index for this actor
    for i in range(num_actors):
        test_actor = gym.get_actor_handle(env, i)
        if test_actor == actor:
            break
        rb_index += gym.get_actor_rigid_body_count(env, test_actor)
    else:
        return False
    
    if rb_index < len(rb_states):
        rb_states[rb_index]['pose']['p'][0] = position.x
        rb_states[rb_index]['pose']['p'][1] = position.y
        rb_states[rb_index]['pose']['p'][2] = position.z
        
        rb_states[rb_index]['pose']['r'][0] = rotation.x
        rb_states[rb_index]['pose']['r'][1] = rotation.y
        rb_states[rb_index]['pose']['r'][2] = rotation.z
        rb_states[rb_index]['pose']['r'][3] = rotation.w
        
        rb_states[rb_index]['vel']['linear'][0] = 0.0
        rb_states[rb_index]['vel']['linear'][1] = 0.0
        rb_states[rb_index]['vel']['linear'][2] = 0.0
        rb_states[rb_index]['vel']['angular'][0] = 0.0
        rb_states[rb_index]['vel']['angular'][1] = 0.0
        rb_states[rb_index]['vel']['angular'][2] = 0.0
        gym.set_sim_rigid_body_states(sim, rb_states, gymapi.STATE_ALL)
        return True
    
    return False


def generate_object_mapping(selected_objects):
    """Generate object mapping dynamically from selected objects"""
    return {f"{obj_name}_base": obj_name for obj_name in selected_objects.keys()}


def generate_joint_name_mapping():
    """Generate joint name mapping using standard patterns"""
    return {
        "drawer_part1": "drawer_part1_joint", 
        "drawer_part2": "drawer_part2_joint", 
        "refrigerator_part1": "refrigerator_part1_joint", 
        "refrigerator_part2": "refrigerator_part2_joint",
        "gasstove_part1": "gasstove_part1_joint", 
        "gasstove_part2": "gasstove_part2_joint",
        "washingmachine_part1": "washingmachine_part1_joint", 
        "microwave_part1": "microwave_part1_joint",
        "sink_part1": "sink_part1_joint", 
        "sink_part2": "sink_part2_joint",
        "laptop_part1": "laptop_part1_joint", 
        "trashbin_part1": "trashbin_part1_joint"
    }


def get_object_colors():
    """Get predefined object colors"""
    return {
        "cup": gymapi.Vec3(0.8, 0.4, 0.4), 
        "bowl": gymapi.Vec3(0.4, 0.8, 0.4), 
        "book": gymapi.Vec3(0.4, 0.4, 0.8),
        "kettle": gymapi.Vec3(0.8, 0.8, 0.4), 
        "pot": gymapi.Vec3(0.8, 0.4, 0.8), 
        "potlid": gymapi.Vec3(0.4, 0.8, 0.8),
        "pan": gymapi.Vec3(0.6, 0.6, 0.4), 
        "knife": gymapi.Vec3(0.8, 0.6, 0.4), 
        "salt": gymapi.Vec3(0.6, 0.8, 0.4),
        "chair": gymapi.Vec3(0.8, 0.4, 0.6), 
        "desk": gymapi.Vec3(0.4, 0.6, 0.8), 
        "bookshelf": gymapi.Vec3(0.6, 0.4, 0.8),
        "diningtable": gymapi.Vec3(0.8, 0.6, 0.6), 
        "laptop": gymapi.Vec3(0.6, 0.8, 0.6), 
        "refrigerator": gymapi.Vec3(0.6, 0.6, 0.8),
        "trashbin": gymapi.Vec3(0.7, 0.7, 0.4), 
        "gasstove": gymapi.Vec3(0.7, 0.4, 0.7), 
        "washingmachine": gymapi.Vec3(0.4, 0.7, 0.7),
        "microwave": gymapi.Vec3(0.8, 0.5, 0.5), 
        "sink": gymapi.Vec3(0.5, 0.8, 0.5), 
        "drawer": gymapi.Vec3(0.5, 0.5, 0.8),
        "cuttingboard": gymapi.Vec3(1.0, 0.6, 0.4)
    }


def get_body_joint_names():
    """Get ordered list of body joint names"""
    return [
        'L_Hip', 'R_Hip', 'Torso',
        'L_Knee', 'R_Knee', 'Spine',
        'L_Ankle', 'R_Ankle', 'Chest',
        'L_Toe', 'R_Toe', 'Neck',
        'L_Thorax', 'R_Thorax', 'Head',
        'L_Shoulder', 'R_Shoulder',
        'L_Elbow', 'R_Elbow',
        'L_Wrist', 'R_Wrist'
    ]


def get_hand_joint_names():
    """Get ordered list of hand joint names"""
    return [
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


def generate_object_transforms_for_frame(object_transforms, object_mapping, selected_objects, frame_idx):
    if not object_transforms or frame_idx >= len(object_transforms):
        return {}
    
    obj_data = object_transforms[frame_idx]
    transforms = {}
    
    for obj_key, obj_name in object_mapping.items():
        if selected_objects[obj_name]:
            position, rotation = matrix_to_gym_transform(obj_data[obj_key])
            transforms[obj_name] = {"pos": position, "rot": rotation}

    return transforms


def get_object_joint_positions_for_frame(object_joint_states, joint_name_mapping, frame_idx):
    if not object_joint_states or frame_idx < 0:
        return {}
    
    frame_joints = {}
    for parahome_joint, urdf_joint in joint_name_mapping.items():
        if parahome_joint in object_joint_states:
            try:
                joint_data = object_joint_states[parahome_joint]
                if frame_idx in joint_data:
                    joint_value = joint_data[frame_idx]
                    if isinstance(joint_value, (int, float)) and np.isfinite(joint_value):
                        frame_joints[urdf_joint] = float(joint_value)
            except Exception:
                continue
    
    return frame_joints


def reset_object_colors(gym, env, object_body_mappings, original_object_colors):
    """Reset all objects to original colors"""
    for obj_name, body_info in object_body_mappings.items():
        if obj_name not in original_object_colors:
            continue
        
        color = original_object_colors[obj_name]
        actor = body_info['actor']
        body_names = body_info['body_names']
        
        for body_idx in range(len(body_names)):
            if len(body_names) > 1 and body_idx > 0:
                part_color = gymapi.Vec3(color.x * 0.8, color.y * 0.8, color.z * 0.8)
                gym.set_rigid_body_color(env, actor, body_idx, gymapi.MESH_VISUAL, part_color)
            else:
                gym.set_rigid_body_color(env, actor, body_idx, gymapi.MESH_VISUAL, color)


def find_body_index(part_name, base_obj_name, body_names, body_dict):
    """Find rigid body index for a part"""
    if part_name == "base":
        return 0
    
    possible_names = [
        part_name, f"{base_obj_name}_{part_name}", part_name.replace('part', ''),
        f"{part_name}_link", f"{base_obj_name}_{part_name}_link"
    ]
    
    for name in possible_names:
        if name in body_dict:
            return body_dict[name]
    
    for i, body_name in enumerate(body_names):
        if part_name in body_name or body_name in part_name:
            return i
    
    return None


def apply_contact_coloring(gym, env, object_body_mappings, original_object_colors, contact_data, frame_idx):
    if not contact_data or not object_body_mappings or frame_idx not in contact_data['sparse_contacts']:
        reset_object_colors(gym, env, object_body_mappings, original_object_colors)
        return
    
    reset_object_colors(gym, env, object_body_mappings, original_object_colors)
    
    frame_contacts = contact_data['sparse_contacts'][frame_idx]
    contact_threshold = 0.01
    red_color = gymapi.Vec3(1.0, 0.0, 0.0)
    
    for obj_part_name, contact_info in frame_contacts.items():
        if '_' in obj_part_name:
            parts = obj_part_name.split('_')
            base_obj_name = parts[0]
            part_name = '_'.join(parts[1:])
        else:
            base_obj_name = obj_part_name
            part_name = "base"
        
        distances = contact_info['distances']
        if len(distances) > 0 and np.any(distances < contact_threshold) and base_obj_name in object_body_mappings:
            body_info = object_body_mappings[base_obj_name]
            actor = body_info['actor']
            body_names = body_info['body_names']
            body_dict = body_info['body_dict']
            
            target_body_idx = find_body_index(part_name, base_obj_name, body_names, body_dict)
            
            if target_body_idx is not None and target_body_idx < len(body_names):
                gym.set_rigid_body_color(env, actor, target_body_idx, gymapi.MESH_VISUAL, red_color)
                if frame_idx < 10:
                    min_dist = np.min(distances)
                    print(f"Frame {frame_idx}: {obj_part_name} -> {body_names[target_body_idx]}[{target_body_idx}] CONTACT - min_dist: {min_dist:.4f}m")
            else:
                gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL, red_color)
                if frame_idx < 10:
                    print(f"Frame {frame_idx}: {obj_part_name} -> BASE[0] CONTACT (part not found) - min_dist: {np.min(distances):.4f}m")

