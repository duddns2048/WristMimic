wandb offline
python wristmimic/run.py \
--task WristMimic_MULTI_OBJ \
--cfg_env wristmimic/data/cfg/parahome_test.yaml \
--cfg_train wristmimic/data/cfg/train/rlg/parahome.yaml \
--test \
--checkpoint checkpoints/<experiment>/nn/<checkpoint>.pth \
--num_envs 1 \
--motion_file data/Parahome/s110_0_kettle_table2desk \
--robot_type sim_human/s110_ROM.xml \
--num_position_iterations 20 \
--num_velocity_iterations 0 \
--test_no_reset