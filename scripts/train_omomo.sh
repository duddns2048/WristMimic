wandb online
python wristmimic/run.py \
--task WristMimic_OMOMO \
--cfg_env wristmimic/data/cfg/omomo_train_new.yaml \
--cfg_train wristmimic/data/cfg/train/rlg/omomo_new.yaml \
--output checkpoints \
--num_envs 2048 \
--minibatch_size 8192 \
--motion_file data/OMOMO/sub1_clothesstand \
--robot_type sim_human/omomo.xml \
--experiment OMOMO_train \
--headless \
--num_position_iterations 20 \
--num_velocity_iterations 0 \