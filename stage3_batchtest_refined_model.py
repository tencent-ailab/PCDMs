import os
import json
import cv2
import torch
from torch import nn
from PIL import Image
import numpy as np
from diffusers import UniPCMultistepScheduler
import torch.nn.functional as F
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPImageProcessor
from src.pipelines.stage3_refined_pipeline import Stage3_RefinedPipeline
import argparse
from transformers import Dinov2Model
from typing import Any, Dict, List, Optional, Tuple, Union
from skimage.metrics import structural_similarity as compare_ssim
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import json
import time
def split_list_into_chunks(lst, n):
    chunk_size = len(lst) // n
    chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    if len(chunks) > n:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)
    return chunks

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module





class ImageProjModel_p(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # b, 257,1280
        return self.net(x)




def inference(args, rank, select_test_datas):

    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)


    save_dir = "{}/show_guidancescale{}_seed{}_numsteps{}/".format(args.save_path, args.guidance_scale, args.seed_number, args.num_inference_steps)
    save_dir_metric = "{}/guidancescale{}_seed{}_numsteps{}/".format(args.save_path, args.guidance_scale, args.seed_number, args.num_inference_steps)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_dir_metric):
        os.makedirs(save_dir_metric, exist_ok=True)

    clip_image_processor = CLIPImageProcessor()

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


    # model define
    image_proj_model_p_dict = {}
    unet_dict = {}

    image_encoder_p = Dinov2Model.from_pretrained(args.image_encoder_p_path).to(device).eval()

    image_proj_model_p = ImageProjModel_p(in_dim=1536, hidden_dim=768, out_dim=1024).to(device).eval()

    model_ckpt = "{}/mp_rank_00_model_states.pt".format(args.weights_name)
    model_sd = torch.load(model_ckpt, map_location="cpu")["module"]

    for k in model_sd.keys():
        if k.startswith("image_proj_model_p"):
            image_proj_model_p_dict[k.replace("image_proj_model_p.", "")] = model_sd[k]

        elif k.startswith("unet"):
            unet_dict[k.replace("unet.", "")] = model_sd[k]

        else:
            print(k)


    image_proj_model_p.load_state_dict(image_proj_model_p_dict)

    pipe = Stage3_RefinedPipeline.from_pretrained(args.pretrained_model_name_or_path,torch_dtype=torch.float16).to(device)

    pipe.unet= UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                           in_channels=8, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(device)

    pipe.unet.load_state_dict(unet_dict)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    print('====================== json_data: {}, model load finish ==================='.format((args.json_path).split('/')[-1]))




    all_ssim = []


    start_time = time.time()
    for data in select_test_datas:

        s_img_path = (args.img_path + data["source_image"].replace('.jpg', '.png'))
        t_img_path = (args.img_path + data["target_image"].replace('.jpg', '.png'))
        gen_t_img_path = args.gen_t_img_path + s_img_path.split("/")[-1].replace(".png", '_to_') + \
                         t_img_path.split("/")[-1]

        s_img = Image.open(s_img_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)
        t_img = Image.open(t_img_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)
        gen_t_img = Image.open(gen_t_img_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)



        clip_processor_s_img = clip_image_processor(images=s_img, return_tensors="pt").pixel_values
        s_img_f = image_encoder_p(clip_processor_s_img.to(device)).last_hidden_state
        s_img_proj_f = image_proj_model_p(s_img_f)  # s_img


        vae_gen_t_image = torch.unsqueeze(img_transform(gen_t_img), 0)





        output = pipe(
                height=args.img_height,
                width=args.img_weigh,
                guidance_rescale=args.guidance_scale,
                vae_gen_t_image=vae_gen_t_image,
                s_img_proj_f=s_img_proj_f,
                num_images_per_prompt=4,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_inference_steps=args.num_inference_steps,
            )


        if args.calculate_metrics:
            ssim_values = []
            for gen_img in output.images:
                ssim_values.append(compare_ssim(np.array(t_img), np.array(gen_img),
                                                gaussian_weights=True, sigma=1.2,
                                                 multichannel=True,channel_axis=2,
                                                ata_range=(np.array(gen_img) * 255.0).max() - (np.array(gen_img) * 255.0).min()
                                                ))

            max_value = max(ssim_values)
            all_ssim.append(max_value)
            max_index = ssim_values.index(max_value)

            grid_metric = output.images[max_index]
            grid_metric.save(save_dir_metric + s_img_path.split("/")[-1].replace(".png", "") + "_to_" +
                                 t_img_path.split("/")[-1])
        else:

            t_pose_path = (args.pose_path + data["target_image"].replace(".jpg", "_pose.jpg"))
            t_pose = Image.open(t_pose_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)

            ssim_values = []
            for gen_img in output.images:
                ssim_values.append(compare_ssim(np.array(t_img), np.array(gen_img),
                                                gaussian_weights=True, sigma=1.2,
                                                 multichannel=True,channel_axis=2,
                                                ata_range=(np.array(gen_img) * 255.0).max() - (np.array(gen_img) * 255.0).min()
                                                ))
            flag_value = min(ssim_values)
            output.images.insert(0, t_img)
            output.images.insert(0, s_img)
            output.images.insert(0, t_pose)
            grid = image_grid(output.images, 1, 7)
            grid.save(
                save_dir + str(flag_value)+"_"+s_img_path.split("/")[-1].replace(".png", "") + "_to_" +
                t_img_path.split("/")[-1])


    end_time =time.time()
    print(end_time-start_time)

    if args.calculate_metrics:
        print(sum(all_ssim)/ len(all_ssim))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple example of a refined model of stage3 script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="/mnt/aigc_cq/private/feishen/weights/stable-diffusion-2-1-base",
                        help="Path to pretrained model or model identifier from huggingface.co/models.", )
    parser.add_argument("--image_encoder_p_path",type=str,default="./dinov2-giant",
        help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--img_path", type=str,default="./datasets/deepfashing/train_all_png/", help="image path", )
    parser.add_argument("--pose_path", type=str,default="./datasets/deepfashing/openpose_all_img/",help="pose path", )
    parser.add_argument("--json_path", type=str,default="./datasets/deepfashing/test_data.json",help="json path", )
    parser.add_argument("--gen_t_img_path", type=str,default="./save_data/stage2/guidancescale2_seed42_numsteps20/",help="gen target image path", )
    parser.add_argument("--save_path", type=str, default="./save_data/stage3", help="save path", )
    parser.add_argument("--guidance_scale",type=int,default=2.0,help="guidance_scale",)
    parser.add_argument("--seed_number",type=int,default=42,help="seed number",)
    parser.add_argument("--num_inference_steps",type=int,default=20,help="num_inference_steps",)
    parser.add_argument("--img_width",type=int,default=512,help="image width",)
    parser.add_argument("--img_height",type=int,default=512,help="image height",)
    parser.add_argument("--calculate_metrics",  action='store_true', help="calculate ssim", )
    parser.add_argument("--weights_name", type=str, default="./Checkpoints/stage2_checkpoints/512",help="weights number", )
    args = parser.parse_args()
    print(args)

    num_devices = torch.cuda.device_count()

    print("using {} num_processes inference".format(num_devices))



    test_data = json.load(open(args.json_path))
    select_test_datas = test_data




    mp.set_start_method("spawn")
    data_list = split_list_into_chunks(select_test_datas, num_devices)


    processes = []
    for rank in range(num_devices):
        p = mp.Process(target=inference, args=(args, rank, data_list[rank] ))
        processes.append(p)
        p.start()

    for rank, p in enumerate(processes):
        p.join()



