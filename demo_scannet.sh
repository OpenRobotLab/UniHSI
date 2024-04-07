python3 unihsi/run.py \
    --task UniHSI_ScanNet\
    --test \
    --num_envs 1 \
    --cfg_env unihsi/data/cfg/humanoid_unified_interaction_scene_1.yaml \
    --cfg_train unihsi/data/cfg/train/rlg/amp_humanoid_task_deep_layer.yaml \
    --motion_file motion_clips/chair_mo014.npy \
    --checkpoint checkpoints/Humanoid.pth \
    --obj_file sceneplan_demo/scannet_example.json