import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import rotation_6d_to_matrix
import pickle
import torch

body_order = {'pHipOrigin': 0,
 'jL5S1': 1,
 'jL4L3': 2,
 'jL1T12': 3,
 'jT9T8': 4,
 'jT1C7': 5,
 'jC1Head': 6,
 'jRightT4Shoulder': 7,
 'jRightShoulder': 8,
 'jRightElbow': 9,
 'jRightWrist': 10,
 'jLeftT4Shoulder': 11,
 'jLeftShoulder': 12,
 'jLeftElbow': 13,
 'jLeftWrist': 14,
 'jRightHip': 15,
 'jRightKnee': 16,
 'jRightAnkle': 17,
 'jRightBallFoot': 18,
 'jLeftHip': 19,
 'jLeftKnee': 20,
 'jLeftAnkle': 21,
 'jLeftBallFoot': 22}

lp_order = {'jLeftWrist': 0, 'jLeftFirstCMC': 1, 'jLeftSecondCMC': 2, 'jLeftThirdCMC': 3, 'jLeftFourthCMC': 4, 'jLeftFifthCMC': 5, 'jLeftFifthMCP': 6, 'jLeftFifthPIP': 7, 'jLeftFifthDIP': 8, 'jLeftFourthMCP': 9, 'jLeftFourthPIP': 10, 'jLeftFourthDIP': 11, 'jLeftThirdMCP': 12, 'jLeftThirdPIP': 13, 'jLeftThirdDIP': 14, 'jLeftSecondMCP': 15, 'jLeftSecondPIP': 16, 'jLeftSecondDIP': 17, 'jLeftFirstMCP': 18, 'jLeftIP': 19}
rp_order = {'jRightWrist': 0, 'jRightFirstCMC': 1, 'jRightSecondCMC': 2, 'jRightThirdCMC': 3, 'jRightFourthCMC': 4, 'jRightFifthCMC': 5, 'jRightFifthMCP': 6, 'jRightFifthPIP': 7, 'jRightFifthDIP': 8, 'jRightFourthMCP': 9, 'jRightFourthPIP': 10, 'jRightFourthDIP': 11, 'jRightThirdMCP': 12, 'jRightThirdPIP': 13, 'jRightThirdDIP': 14, 'jRightSecondMCP': 15, 'jRightSecondPIP': 16, 'jRightSecondDIP': 17, 'jRightFirstMCP': 18, 'jRightIP': 19}

motor_order = {'pHipOrigin': 0, 'jL5S1': 1, 'jL4L3': 2, 'jL1T12': 3, 'jT9T8': 4, 'jT1C7': 5, 'jC1Head': 6, 
               'jRightT4Shoulder': 7, 'jRightShoulder': 8, 'jRightElbow': 9, 'jRightWrist': 10, 
               'jRightFirstCMC': 11, 'jRightFirstMCP': 12, 'jRightIP': 13, 
               'jRightSecondCMC': 14, 'jRightSecondMCP': 15, 'jRightSecondPIP': 16, 'jRightSecondDIP': 17, 
               'jRightThirdCMC': 18, 'jRightThirdMCP': 19, 'jRightThirdPIP': 20, 'jRightThirdDIP': 21, 
               'jRightFourthCMC': 22, 'jRightFourthMCP': 23, 'jRightFourthPIP': 24, 'jRightFourthDIP': 25, 
               'jRightFifthCMC': 26, 'jRightFifthMCP': 27, 'jRightFifthPIP': 28, 'jRightFifthDIP': 29, 
               'jLeftT4Shoulder': 30, 'jLeftShoulder': 31, 'jLeftElbow': 32, 'jLeftWrist': 33, 
               'jLeftFirstCMC': 34, 'jLeftFirstMCP': 35, 'jLeftIP': 36, 
               'jLeftSecondCMC': 37, 'jLeftSecondMCP': 38, 'jLeftSecondPIP': 39, 'jLeftSecondDIP': 40, 
               'jLeftThirdCMC': 41, 'jLeftThirdMCP': 42, 'jLeftThirdPIP': 43, 'jLeftThirdDIP': 44, 
               'jLeftFourthCMC': 45, 'jLeftFourthMCP': 46, 'jLeftFourthPIP': 47, 'jLeftFourthDIP': 48, 
               'jLeftFifthCMC': 49, 'jLeftFifthMCP': 50, 'jLeftFifthPIP': 51, 'jLeftFifthDIP': 52, 
               'jRightHip': 53, 'jRightKnee': 54, 'jRightAnkle': 55, 'jRightBallFoot': 56, 
               'jLeftHip': 57, 'jLeftKnee': 58, 'jLeftAnkle': 59, 'jLeftBallFoot': 60}

def parse_skeleton(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    body_dict = {}

    def parse_body(body_element, parent_name=None):
        name = body_element.get("name")
        pos = np.fromstring(body_element.get("pos"), sep=' ')
        if name not in body_dict:
            body_dict[name] = {"array": pos, "parent": parent_name}
        for child in body_element:
            if child.tag == "body":
                parse_body(child, name)
    
    for body in root.findall(".//body"):
        parse_body(body)
    
    body_dict['pLeftFirstTip'] = {"array": np.array([0.02589792013168335, 0, 0]), "parent": 'jLeftIP'}
    body_dict['pLeftSecondTip'] = {"array": np.array([0.01691189408302307, 0, 0]), "parent": 'jLeftSecondDIP'}
    body_dict['pLeftThirdTip'] = {"array": np.array([0.017722949385643005, 0, 0]), "parent": 'jLeftThirdDIP'}
    body_dict['pLeftFourthTip'] = {"array": np.array([0.01847192645072937, 0, 0]), "parent": 'jLeftFourthDIP'}
    body_dict['pLeftFifthTip'] = {"array": np.array([0.01722918450832367, 0, 0]), "parent": 'jLeftFifthDIP'}
    body_dict['pRightFirstTip'] = {"array": np.array([-0.02589792013168335, 0, 0]), "parent": 'jRightIP'}
    body_dict['pRightSecondTip'] = {"array": np.array([-0.01691189408302307, 0, 0]), "parent": 'jRightSecondDIP'}
    body_dict['pRightThirdTip'] = {"array": np.array([-0.017722949385643005, 0, 0]), "parent": 'jRightThirdDIP'}
    body_dict['pRightFourthTip'] = {"array": np.array([-0.01847192645072937, 0, 0]), "parent": 'jRightFourthDIP'}
    body_dict['pRightFifthTip'] = {"array": np.array([-0.01722918450832367, 0, 0]), "parent": 'jRightFifthDIP'}

    return body_dict


def compute_rotation(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    cross_product = np.cross(vec1, vec2)
    dot_product = np.dot(vec1, vec2)
    
    if np.allclose(cross_product, 0):
        return R.identity()  # Handle the case where vec1 and vec2 are parallel
    
    angle = np.arccos(dot_product)
    rotvec = angle * cross_product / np.linalg.norm(cross_product)
    return R.from_rotvec(rotvec)


def compute_body_local_rotations(body_joint_orientations, local_rotations, joint_order, motor_order, skeleton_hierachy):
    nframes = body_joint_orientations.shape[0]
    wrist_info = {}

    for frame in range(nframes):
        for joint_name, joint_ind in joint_order.items():
            joint_global_rot = rotation_6d_to_matrix(torch.tensor(body_joint_orientations[frame, joint_ind].copy()))
            joint_global_rot = R.from_matrix(joint_global_rot)

            parent_name = get_parent(joint_name, skeleton_hierachy)
            if joint_name in motor_order:
                new_joint_ind = motor_order[joint_name] - 1
                if parent_name is None:
                    local_rotations[frame, new_joint_ind] = joint_global_rot.as_rotvec()
                elif joint_name in ['jLeftWrist', 'jRightWrist']:
                    direction = 'left' if 'Left' in joint_name else 'right'
                    parent_ind = joint_order[parent_name]
                    parent_rot = torch.tensor(body_joint_orientations[frame, parent_ind].copy())
                    parent_rot = R.from_matrix(rotation_6d_to_matrix(parent_rot))
                    if frame not in wrist_info:
                        wrist_info[frame] = {}
                    wrist_info[frame][f'{direction}_wrist_rot'] = joint_global_rot.as_rotvec()
                    wrist_info[frame][f'{direction}_elbow_rot'] = parent_rot.as_rotvec()
                else:
                    parent_name = parent_name.split('Wrist')[0] + 'Wrist' if 'Wrist' in parent_name else parent_name
                    parent_ind = joint_order[parent_name]
                    parent_rot = torch.tensor(body_joint_orientations[frame, parent_ind].copy())
                    parent_rot = R.from_matrix(rotation_6d_to_matrix(parent_rot))
                    joint_local_rot = parent_rot.inv() * joint_global_rot
                    local_rotations[frame, new_joint_ind] = joint_local_rot.as_rotvec()
    
    return local_rotations, wrist_info


def compute_hand_local_rotations(hand_joint_orientations, local_rotations, joint_order, motor_order, skeleton_hierachy, wrist_info):
    nframes = hand_joint_orientations.shape[0]
    for frame in range(nframes):
        for joint_name, joint_ind in joint_order.items():
            joint_global_rot = rotation_6d_to_matrix(torch.tensor(hand_joint_orientations[frame, joint_ind].copy()))
            joint_global_rot = R.from_matrix(joint_global_rot)

            if joint_name in motor_order:
                new_joint_ind = motor_order[joint_name] - 1
                if joint_name in ['jLeftWrist', 'jRightWrist']:
                    direction = 'left' if 'Left' in joint_name else 'right'
                    # get body elbow rotation
                    elbow_rot = wrist_info[frame][f'{direction}_elbow_rot']
                    # get body wrist rotation
                    body_wrist_rot = wrist_info[frame][f'{direction}_wrist_rot']
                    # calculate hand wrist rotation
                    wrist_rot_prime = R.from_rotvec(body_wrist_rot) * joint_global_rot
                    wrist_rot = R.from_rotvec(elbow_rot).inv() * wrist_rot_prime
                    local_rotations[frame, new_joint_ind] = wrist_rot.as_rotvec()
                else:
                    parent_name = get_parent(joint_name, skeleton_hierachy)
                    if 'Wrist' in parent_name:
                        parent_name = parent_name.split('Wrist')[0] + 'Wrist'
                    parent_ind = joint_order[parent_name]
                    parent_rot = torch.tensor(hand_joint_orientations[frame, parent_ind].copy())
                    parent_rot = R.from_matrix(rotation_6d_to_matrix(parent_rot))
                    joint_local_rot = parent_rot.inv() * joint_global_rot
                    local_rotations[frame, new_joint_ind] = joint_local_rot.as_rotvec()
    
    return local_rotations


def get_parent(joint_name, skeleton):
    # Implement logic to find the parent joint name
    # This will depend on your skeleton structure
    parent_name = skeleton[joint_name]['parent']
    return parent_name

def get_child(joint_name, skeleton):
    # Implement logic to find the parent joint name
    # This will depend on your skeleton structure
    childs = []
    for joint in skeleton.keys():
        if skeleton[joint]['parent'] == joint_name:
            childs.append(joint)
    return childs

def calculate_exp_map(A, B):
    # Normalize vectors
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    
    # Calculate rotation axis
    axis = np.cross(A, B)
    
    # Calculate rotation angle
    angle = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))
    
    # Normalize rotation axis
    axis_norm = np.linalg.norm(axis)
    if axis_norm != 0:
        axis /= axis_norm
    
    # Calculate exponential map
    exp_map = axis * angle
    
    return exp_map

def rotate_exp_map_along_xyz(exp_map):
    # 创建初始旋转
    rotation = R.from_rotvec(exp_map)
    # 创建各个轴上的旋转
    rotation_y_90 = R.from_euler('y', 0, degrees=True)
    rotation_z_90 = R.from_euler('z', 0, degrees=True)
    rotation_x_90 = R.from_euler('x', 0, degrees=True)
    # 组合旋转
    combined_rotation = rotation_x_90 * rotation_z_90 * rotation_y_90 * rotation
    # 将组合后的旋转转换回旋转向量
    exp_map = combined_rotation.as_rotvec()
    return exp_map


root_path = 'intermimic/data/assets/parahome/seq/s10'
with open(f'{root_path}/joint_positions.pkl', 'rb') as f:
    joint_positions = pickle.load(f)
with open(f'{root_path}/bone_vectors.pkl', 'rb') as f:
    bone_vectors = pickle.load(f)
with open(f'{root_path}/body_joint_orientations.pkl', 'rb') as f:
    body_joint_orientations = pickle.load(f)
with open(f'{root_path}/hand_joint_orientations.pkl', 'rb') as f:
    hand_joint_orientations = pickle.load(f)


skeleton_hierachy = parse_skeleton('intermimic/data/assets/parahome/sim_raw_human/s10.xml')
local_rotations = np.zeros((body_joint_orientations.shape[0], len(motor_order)-1, 3))  # For exp map
local_rotations, wrist_info = compute_body_local_rotations(body_joint_orientations, local_rotations, body_order, motor_order, skeleton_hierachy) # body
local_rotations = compute_hand_local_rotations(hand_joint_orientations[:,:20], local_rotations, lp_order, motor_order, skeleton_hierachy, wrist_info) # left hand
local_rotations = compute_hand_local_rotations(hand_joint_orientations[:,20:], local_rotations, rp_order, motor_order, skeleton_hierachy, wrist_info) # right hand

np.save(f'{root_path}/local_body_rot.npy', local_rotations)
print(local_rotations.shape)
print("Save local body rotations successfully!")