wandb offline
python intermimic/run.py \
--task InterMimic_OMOMO \
--cfg_env intermimic/data/cfg/omomo_test_new.yaml \
--cfg_train intermimic/data/cfg/train/rlg/omomo_new.yaml \
--test \
--checkpoint checkpoints/OMOMO_train/nn/<checkpoint>.pth \
--num_envs 1 \
--motion_file InterAct/OMOMO/sub1_clothesstand \
--robot_type sim_human/omomo.xml \
--num_position_iterations 20 \
--num_velocity_iterations 0 \
--test_no_reset

# NOTE: --test_no_reset keeps the episode running past a failure: the failure flag
#       is accumulated and the env only resets at the end of the reference sequence.
#       Shared across tasks via Humanoid_MULTI_OBJ_SMPLX._apply_reset().

# Available OMOMO sequences (InterAct/OMOMO/):
#   sub1_clothesstand  sub1_floorlamp  sub1_largetable  sub1_plasticbox
#   sub1_tripod        sub3_monitor    sub3_woodchair   sub4_smallbox
#   sub4_suitcase      sub4_whitechair sub6_smalltable  sub6_trashcan
#   sub7_largebox
