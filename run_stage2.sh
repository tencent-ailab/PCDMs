
accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 --use_deepspeed --mixed_precision="fp16"  stage2_train_inpaint_model.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --image_encoder_p_path='facebook/dinov2-giant' \
  --image_encoder_g_path='{image_encoder_path}' \
  --json_path='{data.json}' \
  --image_root_path="{image_path}"  \
  --output_dir="{output_dir}" \
  --img_height=512  \
  --img_width=512   \
  --learning_rate=1e-4 \
  --train_batch_size=8 \
  --max_train_steps=1000000 \
  --mixed_precision="fp16" \
  --checkpointing_steps=5000  \
  --noise_offset=0.1 \
  --lr_warmup_steps 5000  \
  --seed 42
