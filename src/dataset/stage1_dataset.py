import os
import json
import random
import torch
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor


def read_coordinates_file(file_path):
    coordinates_list = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split())
            coordinates_list.extend([x, y])
    coordinates_tensor = torch.tensor(coordinates_list, dtype=torch.float32).view(1, -1)
    return coordinates_tensor

def PriorCollate_fn(data):
    clip_s_img = torch.stack([example["clip_s_img"] for example in data]).to(memory_format=torch.contiguous_format).float()
    clip_t_img = torch.stack([example["clip_t_img"] for example in data]).to(memory_format=torch.contiguous_format).float()
    s_pose = torch.stack([example["s_pose"] for example in data]).to(memory_format=torch.contiguous_format).float()
    t_pose = torch.stack([example["t_pose"] for example in data]).to(memory_format=torch.contiguous_format).float()

    return {


        "clip_s_img": clip_s_img,
        "clip_t_img": clip_t_img,
        "s_pose": s_pose,
        "t_pose": t_pose,

    }


class PriorImageDataset(Dataset):
    def __init__(
        self,
        json_file,
        size=(512,512),
        s_img_drop_rate=0.0,
        t_img_drop_rate=0.0,
        s_pose_drop_rate=0.0,
        t_pose_drop_rate=0.0,
        image_root_path="",
    ):
        self.data = json.load(open(json_file))

        self.size = size
        self.s_img_drop_rate = s_img_drop_rate
        self.t_img_drop_rate = t_img_drop_rate
        self.s_pose_drop_rate = s_pose_drop_rate
        self.t_pose_drop_rate = t_pose_drop_rate


        self.image_root_path = image_root_path
        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),]
        )


    def __getitem__(self, idx):
        item = self.data[idx]
        # 1. read image
        s_img = Image.open(os.path.join(self.image_root_path, item["source_image"].replace(".jpg", ".png"))).convert("RGB").resize(self.size, Image.BICUBIC)
        t_img = Image.open(os.path.join(self.image_root_path, item["target_image"].replace(".jpg", ".png"))).convert("RGB").resize(self.size, Image.BICUBIC)


        s_pose_path = os.path.join(self.image_root_path, item["source_image"]).replace("/train_all_png/", "/normalized_pose_txt/").replace(".jpg", ".txt")
        t_pose_path = os.path.join(self.image_root_path, item["target_image"]).replace("/train_all_png/", "/normalized_pose_txt/").replace(".jpg", ".txt")
        s_pose = read_coordinates_file(s_pose_path)
        t_pose = read_coordinates_file(t_pose_path)



        # 2. aug
        clip_s_img = (self.clip_image_processor(images=s_img, return_tensors="pt").pixel_values).squeeze(dim=0)
        clip_t_img = (self.clip_image_processor(images=t_img, return_tensors="pt").pixel_values).squeeze(dim=0)


        # 3. drop
        if random.random() < self.s_img_drop_rate:
            clip_s_img = torch.zeros(clip_s_img.shape)

        if  random.random()< self.t_img_drop_rate:
            clip_t_img = torch.zeros(clip_t_img.shape)

        if random.random() < self.s_pose_drop_rate:
            s_pose = torch.zeros(s_pose.shape)

        if random.random() < self.t_pose_drop_rate:
            t_pose = torch.zeros(t_pose.shape)



        return {
            "clip_s_img": clip_s_img,
            "clip_t_img": clip_t_img,
            "s_pose": s_pose,
            "t_pose": t_pose,
        }

    def __len__(self):
        return len(self.data)
