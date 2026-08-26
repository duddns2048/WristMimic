######################################## Data Description ############################################
# bone_vectors: [body 22/lhand 24/rhand 24][njoints] ('pHipOrigin', 'jRightHip'): 0.09663602709770203
# body_global_transform.pkl (nframes, 4, 4)
# body_joint_orientations.pkl (nframes, 23, 6)
# joint_positions.pkl (nframes, 73, 3)
# joint_states.pkl [nobjects](nframes, 1) # 旋转部分用弧度表示(eg. laptop)，棱柱部分用米表示(draw)
# hand_joint_orientations.pkl (nframes, 40, 6)
# head_tips.pkl (nframes, 3)
# object_transformations.pkl [object_name] (nframes, 4, 4)
######################################################################################################

import os
import pickle
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import rotation_6d_to_matrix

start_frame = 4410
end_frame = 4530
obj_names = ['cup']

root_path = 'intermimic/data/assets/parahome/seq/s10'
with open(f'{root_path}/bone_vectors.pkl', 'rb') as f:
    bone_vectors = pickle.load(f)
with open(f'{root_path}/body_global_transform.pkl', 'rb') as f:
    body_global_transform = pickle.load(f)
with open(f'{root_path}/body_joint_orientations.pkl', 'rb') as f:
    body_joint_orientations = pickle.load(f)
with open(f'{root_path}/hand_joint_orientations.pkl', 'rb') as f:
    hand_joint_orientations = pickle.load(f)
with open(f'{root_path}/joint_positions.pkl', 'rb') as f:
    joint_positions = pickle.load(f)
with open(f'{root_path}/object_transformations.pkl', 'rb') as f:
    object_transformations = pickle.load(f)
local_body_rot = np.load(f'{root_path}/local_body_rot.npy')[start_frame:end_frame]

# settings
# 0-22: body motion (23)
# 23-47: left hands (25)
# 48-72: right hands (25)
body_joint_order = {'pHipOrigin': 0, 'jL5S1': 1, 'jL4L3': 2, 'jL1T12': 3, 'jT9T8': 4, 'jT1C7': 5, 'jC1Head': 6, 'jRightT4Shoulder': 7, 'jRightShoulder': 8, 'jRightElbow': 9, 'jRightWrist': 10, 'jLeftT4Shoulder': 11, 'jLeftShoulder': 12, 'jLeftElbow': 13, 'jLeftWrist': 14, 'jRightHip': 15, 'jRightKnee': 16, 'jRightAnkle': 17, 'jRightBallFoot': 18, 'jLeftHip': 19, 'jLeftKnee': 20, 'jLeftAnkle': 21, 'jLeftBallFoot': 22}
lhand_joint_order = {'jLeftWrist': 0, 'jLeftFirstCMC': 1, 'jLeftSecondCMC': 2, 'jLeftThirdCMC': 3, 'jLeftFourthCMC': 4, 'jLeftFifthCMC': 5, 'jLeftFifthMCP': 6, 'jLeftFifthPIP': 7, 'jLeftFifthDIP': 8, 'pLeftFifthTip': 9, 'jLeftFourthMCP': 10, 'jLeftFourthPIP': 11, 'jLeftFourthDIP': 12, 'pLeftFourthTip': 13, 'jLeftThirdMCP': 14, 'jLeftThirdPIP': 15, 'jLeftThirdDIP': 16, 'pLeftThirdTip': 17, 'jLeftSecondMCP': 18, 'jLeftSecondPIP': 19, 'jLeftSecondDIP': 20, 'pLeftSecondTip': 21, 'jLeftFirstMCP': 22, 'jLeftIP': 23, 'pLeftFirstTip': 24}
rhand_joint_order = {'jRightWrist': 0, 'jRightFirstCMC': 1, 'jRightSecondCMC': 2, 'jRightThirdCMC': 3, 'jRightFourthCMC': 4, 'jRightFifthCMC': 5, 'jRightFifthMCP': 6, 'jRightFifthPIP': 7, 'jRightFifthDIP': 8, 'pRightFifthTip': 9, 'jRightFourthMCP': 10, 'jRightFourthPIP': 11, 'jRightFourthDIP': 12, 'pRightFourthTip': 13, 'jRightThirdMCP': 14, 'jRightThirdPIP': 15, 'jRightThirdDIP': 16, 'pRightThirdTip': 17, 'jRightSecondMCP': 18, 'jRightSecondPIP': 19, 'jRightSecondDIP': 20, 'pRightSecondTip': 21, 'jRightFirstMCP': 22, 'jRightIP': 23, 'pRightFirstTip': 24}
# motor_order = {'pHipOrigin': 0, 'jL5S1': 1, 'jL4L3': 2, 'jL1T12': 3, 'jT9T8': 4, 'jT1C7': 5, 'jC1Head': 6, 
#                'jRightT4Shoulder': 7, 'jRightShoulder': 8, 'jRightElbow': 9, 'jRightWrist': 10, 
#                'jRightWrist1': 11, 'jRightFirstCMC': 12, 'jRightFirstMCP': 13, 'jRightIP': 14, 
#                'jRightWrist2': 15, 'jRightSecondCMC': 16, 'jRightSecondMCP': 17, 'jRightSecondPIP': 18, 'jRightSecondDIP': 19, 
#                'jRightWrist3': 20, 'jRightThirdC360MC': 21, 'jRightThirdMCP': 22, 'jRightThirdPIP': 23, 'jRightThirdDIP': 24, 
#                'jRightWrist4': 25, 'jRightFourthCMC': 26, 'jRightFourthMCP': 27, 'jRightFourthPIP': 28, 'jRightFourthDIP': 29, 
#                'jRightWrist5': 30, 'jRightFifthCMC': 31, 'jRightFifthMCP': 32, 'jRightFifthPIP': 33, 'jRightFifthDIP': 34, 
#                'jLeftT4Shoulder': 35, 'jLeftShoulder': 36, 'jLeftElbow': 37, 'jLeftWrist': 38, 
#                'jLeftWrist1': 39, 'jLeftFirstCMC': 40, 'jLeftFirstMCP': 41, 'jLeftIP': 42, 
#                'jLeftWrist2': 43, 'jLeftSecondCMC': 44, 'jLeftSecondMCP': 45, 'jLeftSecondPIP': 46, 'jLeftSecondDIP': 47, 
#                'jLeftWrist3': 48, 'jLeftThirdCMC': 49, 'jLeftTrhand_joint_orderhirdMCP': 50, 'jLeftThirdPIP': 51, 'jLeftThirdDIP': 52, 
#                'jLeftWrist4': 53, 'jLeftFourthCMC': 54, 'jLeftFourthMCP': 55, 'jLeftFourthPIP': 56, 'jLeftFourthDIP': 57, 
#                'jLeftWrist5': 58, 'jLeftFifthCMC': 59, 'jLeftFifthMCP': 60, 'jLeftFifthPIP': 61, 'jLeftFifthDIP': 62, 
#                'jRightHip': 63, 'jRightKnee': 64, 'jRightAnkle': 65, 'jRightBallFoot': 66, 'jLeftHip': 67, 
#                'jLeftKnee': 68, 'jLeftAnkle': 69, 'jLeftBallFoot': 70}
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
num_motors = len(motor_order) - 1
num_joints = 71

# human
# root_pos(3) + root_rot(3) + root_pos_vel(3) +dof_pos(60*3) + joint_pos(71*3) + obj_pos(3) + obj_rot(3) + contact_graph(1)
motion_dim = 6 + 3 + num_motors*3 + num_joints*3 + 6*len(obj_names) + 1 
motion = torch.zeros(end_frame-start_frame, motion_dim)
root_pos = np.stack([body_global_transform[i][:3, 3] for i in range(start_frame, end_frame)])
motion[:, 0:3] = torch.tensor(root_pos) # root_pos
root_rot = R.from_matrix([body_global_transform[i][:3, :3] for i in range(start_frame, end_frame)])
motion[:, 3:6] = torch.tensor(root_rot.as_rotvec()) # root_rot_3d
start_ind = 9
motion[:, start_ind:start_ind+num_motors*3] = torch.tensor(local_body_rot).view(-1, num_motors*3) # dof_pos

joint_pos = torch.zeros(end_frame-start_frame, num_joints*3)
joint_pos[:,0:3] = torch.tensor(joint_positions[start_frame:end_frame, 0]) # pHipOrigin
for key in motor_order:
    ind = motor_order[key]
    if key in body_joint_order:
        joint_pos[:, 3*ind:3*ind+3] = torch.tensor(joint_positions[start_frame:end_frame, body_joint_order[key]])
    elif key in lhand_joint_order:
        joint_pos[:, 3*ind:3*ind+3] = torch.tensor(joint_positions[start_frame:end_frame, 23+lhand_joint_order[key]])
    elif key in rhand_joint_order:
        joint_pos[:, 3*ind:3*ind+3] = torch.tensor(joint_positions[start_frame:end_frame, 48+rhand_joint_order[key]])
start_ind += num_motors * 3
motion[:, start_ind:start_ind+num_joints*3] = joint_pos

# obejct
start_ind += num_joints*3
for obj_name in obj_names:
    obj_pos = np.stack([object_transformations[i][f'{obj_name}_base'][:3, 3] for i in range(start_frame, end_frame)])
    obj_rot = R.from_matrix([object_transformations[i][f'{obj_name}_base'][:3, :3] for i in range(start_frame, end_frame)])
    motion[:, start_ind:start_ind+3] = torch.tensor(obj_pos) # obj_pos
    start_ind += 3
    motion[:, start_ind:start_ind+3] = torch.tensor(obj_rot.as_rotvec()) # obj_rot
    start_ind += 3


# # contact information
# contact_frames = [_ for _ in range(1877, 2036)]
# contact_graph = torch.zeros(end_frame-start_frame, 1)
# for i in range(start_frame, end_frame):
#     contact_graph[i-start_frame] = 1 if i in contact_frames else 0
# start_ind += 3
# motion[:, start_ind:start_ind+1] = contact_graph

# motion = motion.repeat(3,1)

# save into pt file
motion_name = f's10.tmp'

save_path = f'tmp/{motion_name}.pt'

os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(motion.to('cuda'), save_path)
print(f'Save motion into {save_path}')
print(f'Motion shape: {motion.shape}')