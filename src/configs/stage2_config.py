import argparse
parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",)


parser.add_argument(
    "--seed", type=int, default=None, help="A seed for reproducible training."
)

parser.add_argument(
    "--train_batch_size",
    type=int,
    default=4,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument("--num_train_epochs", type=int, default=10000)
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
)
parser.add_argument(
    "--checkpointing_steps",
    type=int,
    default=500,
    help=(
        "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
        "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
        "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
        "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
        "instructions."
    ),
)
parser.add_argument("--json_path", type=str, default="./datasets/deepfashing/train_data.json", help="json path", )
parser.add_argument("--image_root_path", type=str, default="./datasets/deepfashing/all_data_png/", help="image path", )
parser.add_argument("--image_encoder_g_path", type=str, default="./OpenCLIP-ViT-H-14",
                    help="Path to pretrained model or model identifier from huggingface.co/models.", )
parser.add_argument("--image_encoder_p_path", type=str, default="./dinov2-giant",
                    help="Path to pretrained model or model identifier from huggingface.co/models.", )
parser.add_argument("--output_dir",type=str,default="./logs/stage2",help="The output directory where the model predictions and checkpoints will be written.",)
parser.add_argument("--img_width", type=int, default=512, help="device number", )
parser.add_argument("--img_height", type=int, default=512, help="device number", )
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help=(
        "Whether training should be resumed from a previous checkpoint. Use a path saved by"
        ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
    ),
)
parser.add_argument(
    "--set_grads_to_none",
    action="store_true",
    help=(
        "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
        " behaviors, so disable this argument if it causes any problems. More info:"
        " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
    ),
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-6,
    help="Initial learning rate (after the potential warmup period) to use.",
)

parser.add_argument(
    "--scale_lr",
    action="store_true",
    default=False,
    help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
)
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="constant",
    help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
    ),
)
parser.add_argument(
    "--lr_warmup_steps",
    type=int,
    default=500,
    help="Number of steps for the warmup in the lr scheduler.",
)
parser.add_argument(
    "--lr_num_cycles",
    type=int,
    default=1,
    help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
)
parser.add_argument(
    "--lr_power",
    type=float,
    default=1.0,
    help="Power factor of the polynomial scheduler.",
)


parser.add_argument(
    "--adam_beta1",
    type=float,
    default=0.9,
    help="The beta1 parameter for the Adam optimizer.",
)
parser.add_argument(
    "--adam_beta2",
    type=float,
    default=0.999,
    help="The beta2 parameter for the Adam optimizer.",
)
parser.add_argument(
    "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
)
parser.add_argument(
    "--adam_epsilon",
    type=float,
    default=1e-08,
    help="Epsilon value for the Adam optimizer",
)
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
)
parser.add_argument(
    "--logging_dir",
    type=str,
    default="logs",
    help=(
        "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
        " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    ),
)
parser.add_argument(
    "--report_to",
    type=str,
    default="tensorboard",
    help=(
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
        ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
        "Only applicable when `--with_tracking` is passed."
    ),
)
parser.add_argument(
    "--allow_tf32",
    action="store_true",
    help=(
        "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
        " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    ),
)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default=None,
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)



parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")




args = parser.parse_args()
print(args)



