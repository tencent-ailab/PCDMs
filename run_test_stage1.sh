python3  stage1_batchtest_prior_model.py \
 --pretrained_model_name_or_path="kandinsky-community/kandinsky-2-2-prior" \
 --image_encoder_path="{image_encoder_path}" \
 --img_path='{image_path}' \
 --json_path='{data.json}' \
 --pose_path="{normalized_pose_txt}" \
 --save_path="./logs/view_stage1/512_512" \
 --weights_name="{save_ckpt}"\

