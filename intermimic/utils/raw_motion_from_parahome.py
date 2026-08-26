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

start_frame = 2580
end_frame = 2680
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

# lhand_motor_order = {'jLeftWrist': 0, 'jLeftFirstCMC': 1, 'jLeftSecondCMC': 2, 'jLeftThirdCMC': 3, 'jLeftFourthCMC': 4, 
#                     'jLeftFifthCMC': 5, 'jLeftFifthMCP': 6, 'jLeftFifthPIP': 7, 'jLeftFifthDIP': 8, 'jLeftFourthMCP': 9, 'jLeftFourthPIP': 10, 
#                     'jLeftFourthDIP': 11, 'jLeftThirdMCP': 12, 'jLeftThirdPIP': 13, 'jLeftThirdDIP': 14, 'jLeftSecondMCP': 15, 
#                     'jLeftSecondPIP': 16, 'jLeftSecondDIP': 17, 'jLeftFirstMCP': 18, 'jLeftIP': 19}
# rhand_motor_order = {'jRightWrist': 0, 'jRightFirstCMC': 1, 'jRightSecondCMC': 2, 'jRightThirdCMC': 3, 'jRightFourthCMC': 4, 
#                     'jRightFifthCMC': 5, 'jRightFifthMCP': 6, 'jRightFifthPIP': 7, 'jRightFifthDIP': 8, 'jRightFourthMCP': 9, 'jRightFourthPIP': 10, 
#                     'jRightFourthDIP': 11, 'jRightThirdMCP': 12, 'jRightThirdPIP': 13, 'jRightThirdDIP': 14, 'jRightSecondMCP': 15, 
#                     'jRightSecondPIP': 16, 'jRightSecondDIP': 17, 'jRightFirstMCP': 18, 'jRightIP': 19}

# settings
# 0-22: body motion (23)
# 23-47: left hands (25)
# 48-72: right hands (25)
body_joint_order = {'pHipOrigin': 0, 'jL5S1': 1, 'jL4L3': 2, 'jL1T12': 3, 'jT9T8': 4, 'jT1C7': 5, 'jC1Head': 6, 
                    'jRightT4Shoulder': 7, 'jRightShoulder': 8, 'jRightElbow': 9, 'jRightWrist': 10, 
                    'jLeftT4Shoulder': 11, 'jLeftShoulder': 12, 'jLeftElbow': 13, 'jLeftWrist': 14, 
                    'jRightHip': 15, 'jRightKnee': 16, 'jRightAnkle': 17, 'jRightBallFoot': 18, 
                    'jLeftHip': 19, 'jLeftKnee': 20, 'jLeftAnkle': 21, 'jLeftBallFoot': 22}
lhand_joint_order = {'jLeftWrist': 0, 'jLeftFirstCMC': 1, 'jLeftSecondCMC': 2, 'jLeftThirdCMC': 3, 'jLeftFourthCMC': 4, 'jLeftFifthCMC': 5, 
                    'jLeftFifthMCP': 6, 'jLeftFifthPIP': 7, 'jLeftFifthDIP': 8, 'pLeftFifthTip': 9, 'jLeftFourthMCP': 10, 'jLeftFourthPIP': 11,
                    'jLeftFourthDIP': 12, 'pLeftFourthTip': 13, 'jLeftThirdMCP': 14, 'jLeftThirdPIP': 15, 'jLeftThirdDIP': 16,
                    'pLeftThirdTip': 17, 'jLeftSecondMCP': 18, 'jLeftSecondPIP': 19, 'jLeftSecondDIP': 20, 'pLeftSecondTip': 21, 
                    'jLeftFirstMCP': 22, 'jLeftIP': 23, 'pLeftFirstTip': 24}
rhand_joint_order = {'jRightWrist': 0, 'jRightFirstCMC': 1, 'jRightSecondCMC': 2, 'jRightThirdCMC': 3, 'jRightFourthCMC': 4, 'jRightFifthCMC': 5,
                    'jRightFifthMCP': 6, 'jRightFifthPIP': 7, 'jRightFifthDIP': 8, 'pRightFifthTip': 9, 'jRightFourthMCP': 10, 'jRightFourthPIP': 11, 
                    'jRightFourthDIP': 12, 'pRightFourthTip': 13, 'jRightThirdMCP': 14, 'jRightThirdPIP': 15, 'jRightThirdDIP': 16, 
                    'pRightThirdTip': 17, 'jRightSecondMCP': 18, 'jRightSecondPIP': 19, 'jRightSecondDIP': 20, 'pRightSecondTip': 21, 
                    'jRightFirstMCP': 22, 'jRightIP': 23, 'pRightFirstTip': 24}
                

# 73 - 10 fingertips - 2 overlapping joints
# 61 joints
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


# Without pHipOrigin
num_motors = len(motor_order) - 1
num_joints = 71
n_frames = end_frame - start_frame

# root pos(3) + root rot(4) + 2 + dof_pos(60*3) + joint position(71*3, end effector + pelvis) + joint rotation(71*4)
motion_dim = 3 + 4 + 2 + num_motors*3 + num_joints*3 + num_joints*4
motion = torch.zeros(n_frames, motion_dim)

# root position 3
root_pos = np.stack([body_global_transform[i][:3, 3] for i in range(start_frame, end_frame)])
motion[:, 0:3] = torch.tensor(root_pos)

# root rotation 4
root_rot = R.from_matrix([body_global_transform[i][:3, :3] for i in range(start_frame, end_frame)])
motion[:, 3:7] = torch.tensor(root_rot.as_quat())

# dof_pose 60*3
start_ind = 9
motion[:, start_ind:start_ind+num_motors*3] = torch.tensor(local_body_rot).view(-1, num_motors*3) # dof_pos

# joint position 71*3 / last 10 fingertips, not used
joint_pos = torch.zeros(n_frames, num_joints*3)
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


# joint rotation 71*4
root_quat = root_rot.as_quat()                               # (n_frame, 4) [x,y,z,w]
root_quat_rep = np.repeat(root_quat[:, None, :], num_joints, axis=1)  # (n_frame, J, 4)
root_rot_rep = R.from_quat(root_quat_rep.reshape(-1, 4))     # (n_frame*J,)

local_rot = R.from_rotvec(local_body_rot.reshape(-1, 3)) # (n_frame, 60, 3) -> (n_frame * 60, 3)
local_rot = R.as_quat(local_rot).reshape(n_frames, num_motors, 4)

pad_start = np.tile(np.array([0, 0, 0, 1]), (n_frames, 1, 1))
pad_end = np.tile(np.array([0, 0, 0, 1]), (n_frames, 10, 1))
local_rot = np.concatenate([pad_start, local_rot, pad_end], axis=1).reshape(-1, 4)
local_rot = R.from_quat(local_rot.reshape(-1, 4)) 

global_joint_rot = root_rot_rep * local_rot # (n_frame*J,)
global_joint_quat = global_joint_rot.as_quat().reshape(n_frames, num_joints, 4)  # [x,y,z,w]
start_ind += num_joints * 3
motion[:, start_ind:start_ind+num_joints*4] = torch.tensor(global_joint_quat.reshape(n_frames, -1))


smplx_motion_path = 'InterAct/ParaHome_s10/s10_0_kettle,diningtable,sink_100.pt'
smplx_motion = torch.load(smplx_motion_path)
object_motion = smplx_motion['data'][:, 526:]
torch.save(torch.cat([motion, object_motion], dim=1), ('InterAct/ParaHome_s10_raw/s10_0_kettle,diningtable,sink_100.pt'))