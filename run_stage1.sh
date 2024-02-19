accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 --use_deepspeed --num_processes 8   \
  stage1_train_prior_model.py \
  --pretrained_model_name_or_path="kandinsky-community/kandinsky-2-2-prior" \
  --image_encoder_path='{image_encoder_path}' \
  --img_path='{image_path}' \
  --json_path='{data.json}' \
  --output_dir="{output_dir}" \
  --img_height=512  \
  --img_width=512   \
  --train_batch_size=128 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100000 \
  --noise_offset=0.1 \
  --learning_rate=1e-05 \
  --weight_decay=0.01 \
  --lr_scheduler="constant" --num_warmup_steps=2000 \
  --checkpointing_steps=5000 \
  --seed 42