wandb online
python intermimic/run.py \
--task InterMimic_OMOMO \
--cfg_env intermimic/data/cfg/omomo_train_new.yaml \
--cfg_train intermimic/data/cfg/train/rlg/omomo_new.yaml \
--output checkpoints \
--num_envs 2048 \
--minibatch_size 8192 \
--motion_file InterAct/OMOMO/sub1_clothesstand \
--robot_type sim_human/omomo.xml \
--experiment OMOMO_train \
--headless \
--num_position_iterations 20 \
--num_velocity_iterations 0 \