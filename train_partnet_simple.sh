python3 unihsi/run.py \
    --task UniHSI_PartNet_Train\
    --cfg_env unihsi/data/cfg/humanoid_unified_interaction_scene_64.yaml \
    --cfg_train unihsi/data/cfg/train/rlg/amp_humanoid_task_deep_layer_2we.yaml \
    --motion_file motion_clips/training.yaml \
    --output_path output/ \
    --headless \
    --obj_file sceneplan/partnet_train_simple.json