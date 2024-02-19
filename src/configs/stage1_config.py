

import os
import argparse




parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--pretrained_image_model_path",
    type=str,
    default=None,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--pretrained_pose_model_path",
    type=str,
    default=None,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)

parser.add_argument(
    "--unet_config_file",
    type=str,
    default=None,
    help="Config file of UNet model",
)
parser.add_argument("--json_path", type=str, default="./datasets/deepfashing/train_data.json", help="json path", )
parser.add_argument("--img_path", type=str, default="./datasets/deepfashing/all_data_png/", help="image path", )
parser.add_argument("--image_encoder_path", type=str, default="./OpenCLIP-ViT-H-14",
                    help="Path to pretrained model or model identifier from huggingface.co/models.", )
parser.add_argument("--img_width", type=int, default=512, help="width", )
parser.add_argument("--img_height", type=int, default=512, help="height", )
parser.add_argument(
    "--output_dir",
    type=str,
    default="sd-model-finetuned",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

parser.add_argument(
    "--center_crop",
    action="store_true",
    help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
)
parser.add_argument(
    "--random_flip",
    action="store_true",
    help="whether to randomly flip images horizontally",
)
parser.add_argument(
    '--clip_penultimate',
    type=bool,
    default=False,
    help='Use penultimate CLIP layer for text embedding'
)
parser.add_argument(
    "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
)
parser.add_argument("--num_train_epochs", type=int, default=100000000)
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
    default=1e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.01,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="linear",
    help="The scheduler type to use.",
    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
)
parser.add_argument(
    "--num_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
)
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
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
    "--print_freq",
    type=int,
    default=1,
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
    "--checkpointing_steps",
    type=int,
    default=500,
    help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
)
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help="If the training should continue from a checkpoint folder.",
)
parser.add_argument(
    "--unet_init_ckpt",
    type=str,
    default=None,
    help="If the training should continue from a checkpoint folder.",
)

parser.add_argument(
    "--mixed_precision",
    type=str,
    default="fp16",
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)
parser.add_argument(
    "--enable_xformers_memory_efficient_attention",
    action="store_true",
    help="Whether or not to use xformers.",
)
parser.add_argument(
    "--max_grad_norm", default=10.0, type=float, help="Max gradient norm."
)
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")

args = parser.parse_args()
print(args)

