python3  stage3_batchtest_refined_model.py \
  --img_weigh 512 \
  --img_height 512 \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
 --image_encoder_p_path='facebook/dinov2-giant' \
 --img_path='{image_path}' \
 --json_path='{data.json}' \
 --pose_path="{pose_path}" \
 --gen_t_img_path="./logs/view_stage2/512_512/" \
 --save_path="./logs/view_stage3/512_512" \
 --weights_name"{save_ckpt}" \
 --calculate_metrics