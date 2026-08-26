wandb offline
python intermimic/run.py \
--task InterMimic_MULTI_OBJ \
--cfg_env intermimic/data/cfg/parahome_test.yaml \
--cfg_train intermimic/data/cfg/train/rlg/parahome.yaml \
--test \
--checkpoint checkpoints/<experiment>/nn/<checkpoint>.pth \
--num_envs 1 \
--motion_file InterAct/Parahome/s110_0_kettle_table2desk \
--robot_type sim_human/s110_ROM.xml \
--num_position_iterations 20 \
--num_velocity_iterations 0 \
--test_no_reset