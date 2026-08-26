wandb online
python intermimic/run.py \
--task InterMimic_MULTI_OBJ \
--cfg_env intermimic/data/cfg/parahome_train.yaml \
--cfg_train intermimic/data/cfg/train/rlg/parahome.yaml \
--output checkpoints \
--num_envs 2048 \
--minibatch_size 16384 \
--motion_file InterAct/Parahome/s110_0_kettle_table2desk \
--robot_type sim_human/s110_ROM.xml \
--experiment parahand_s110_0_kettle_table2desk \
--headless \
--num_position_iterations 20 \
--num_velocity_iterations 0 \
# --resume_from checkpoints/s110/nn/exp225_00043500.pth

# NOTE: All grasp/reset timing, thresholds, and weights are now configured in:
#       intermimic/data/cfg/parahome_train.yaml
# Previously overriqqdden parameters (now removed):
#   --grasp_window_before, --grasp_window_after
#   --reset_transition_frame (now resetTransitionFrame1 and resetTransitionFrame2)
#   --wrist_pos_reset_threshold1/2/3, --wrist_rot_reset_threshold1/2/3
#   --finger_window_before, --finger_window_after
#   --reset_window_before

# 2048 8192
# add_surfacenormal 