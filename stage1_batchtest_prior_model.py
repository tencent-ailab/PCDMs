import os
import torch
from PIL import Image
from src.models.stage1_prior_transformer import Stage1_PriorTransformer
from src.pipelines.stage1_prior_pipeline import Stage1_PriorPipeline
import torch.nn.functional as F
from transformers import (
    CLIPVisionModelWithProjection,

    CLIPImageProcessor,
)
import argparse
import numpy as np

import torch.multiprocessing as mp
import json
import time


# Read a text file and convert the coordinates into a tensor
def read_coordinates_file(file_path):
    coordinates_list = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split())
            coordinates_list.extend([x, y])
    coordinates_tensor = torch.tensor(coordinates_list, dtype=torch.float32).view(1, -1)
    return coordinates_tensor

def split_list_into_chunks(lst, n):
    chunk_size = len(lst) // n
    chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    if len(chunks) > n:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)
    return chunks

def main(args, rank, select_test_datas,):

    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)

    # save path
    save_dir = "{}/guidancescale{}_seed{}_numsteps{}/".format(args.save_path,  args.guidance_scale, args.seed_number, args.num_inference_steps)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # prepare data aug
    clip_image_processor = CLIPImageProcessor()

    # prepare model
    model_ckpt = "{}/mp_rank_00_model_states.pt".format(args.weights_name)


    pipe = Stage1_PriorPipeline.from_pretrained(args.pretrained_model_name_or_path).to(device)
    pipe.prior= Stage1_PriorTransformer.from_pretrained(args.pretrained_model_name_or_path, subfolder="prior", num_embeddings=2,embedding_dim=1024, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(device)

    prior_dict = torch.load(model_ckpt, map_location="cpu")["module"]
    pipe.prior.load_state_dict(prior_dict)
    pipe.enable_xformers_memory_efficient_attention()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).eval().to(device)


    print('====================== model load finish ===================')


    # start test
    start_time = time.time()
    number = 0
    sum_simm = 0
    for select_test_data in select_test_datas:
        number += 1

        #prepare data
        s_img_path = (args.img_path + select_test_data["source_image"].replace('.jpg', '.png'))
        t_img_path = (args.img_path + select_test_data["target_image"].replace('.jpg', '.png'))

        s_pose_path = args.pose_path + select_test_data['source_image'].replace('.jpg', '.txt')
        t_pose_path = (args.pose_path + select_test_data["target_image"].replace(".jpg", ".txt"))

        # image_pair
        s_image = Image.open(s_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)
        t_image = Image.open(t_img_path).convert("RGB").resize((args.img_width, args.img_height), Image.BICUBIC)

        s_pose = read_coordinates_file(s_pose_path).to(device).unsqueeze(1)
        t_pose = read_coordinates_file(t_pose_path).to(device).unsqueeze(1)





        clip_s_image = clip_image_processor(images=s_image, return_tensors="pt").pixel_values
        clip_t_image = clip_image_processor(images=t_image, return_tensors="pt").pixel_values


        with torch.no_grad():
            s_img_embed = (image_encoder(clip_s_image.to(device)).image_embeds).unsqueeze(1)
            target_embed = image_encoder(clip_t_image.to(device)).image_embeds




        output = pipe(
            s_embed = s_img_embed,
            s_pose = s_pose,
            t_pose = t_pose,
            num_images_per_prompt=1,
            num_inference_steps = args.num_inference_steps,
            generator = generator,
            guidance_scale = args.guidance_scale,
        )

        # save features
        feature = output[0].cpu().detach().numpy()
        np.save(save_dir + s_img_path.split("/")[-1].replace(".png", '_to_') + t_img_path.split("/")[-1].replace(".png", '.npy'),  feature)

        # computer scores
        predict_embed = output[0]

        cosine_similarities = F.cosine_similarity(predict_embed, target_embed)
        sum_simm += cosine_similarities.item()

    end_time =time.time()
    print(end_time-start_time)


    avg_simm = sum_simm/number
    with open (save_dir+'/a_results.txt', 'a') as ff:
        ff.write('number is {}, guidance_scale is {}, all averge simm is :{} \n'.format(number, args.guidance_scale, avg_simm))
    print('number is {}, guidance_scale is {}, all averge simm is :{}'.format(number, args.guidance_scale, avg_simm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a prior model of stage1 script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="./kandinsky-2-2-prior",
        help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--image_encoder_path",type=str,default="./OpenCLIP-ViT-H-14",
        help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--img_path", type=str, default="./datasets/deepfashing/train_all_png/", help="image path", )
    parser.add_argument("--pose_path", type=str, default="./datasets/deepfashing/normalized_pose_txt/", help="pose path", )
    parser.add_argument("--json_path", type=str, default="./datasets/deepfashing/test_data.json", help="json path", )
    parser.add_argument("--save_path", type=str, default="./save_data/stage1", help="save path", )
    parser.add_argument("--guidance_scale",type=int,default=0,help="guidance_scale",)
    parser.add_argument("--seed_number",type=int,default=42,help="seed number",)
    parser.add_argument("--num_inference_steps",type=int,default=20,help="num_inference_steps",)
    parser.add_argument("--img_width",type=int,default=512,help="image width",)
    parser.add_argument("--img_height",type=int,default=512,help="image height",)
    parser.add_argument("--weights_name",type=str,default="./Checkpoints/stage1_checkpoints/512",help="weights number",)


    args = parser.parse_args()
    print(args)

    # Set the number of GPUs.
    num_devices = torch.cuda.device_count()

    print("Using {} GPUs inference".format(num_devices))

    # load data
    test_data = json.load(open(args.json_path))
    select_test_datas = test_data
    print('The number of test data: {}'.format(len(select_test_datas)))

    # Create a process pool
    mp.set_start_method("spawn")
    data_list = split_list_into_chunks(select_test_datas, num_devices)

    processes = []
    for rank in range(num_devices):
        p = mp.Process(target=main, args=(args,rank, data_list[rank], ))
        processes.append(p)
        p.start()


    for rank, p in enumerate(processes):
        p.join()







