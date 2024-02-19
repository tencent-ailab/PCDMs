import os
import json
import random
import torch
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler
from torchvision import transforms
from PIL import Image
from transformers import (
    CLIPImageProcessor,
)




def RefinedCollate_fn(data):


    source_image = torch.stack([example["clip_s_img"] for example in data])
    source_image = source_image.to(memory_format=torch.contiguous_format).float()


    target_image = torch.stack([example["trans_t_img"] for example in data])
    target_image = target_image.to(memory_format=torch.contiguous_format).float()



    gen_target_image = torch.stack([example["trans_gen_t_img"] for example in data])
    gen_target_image = gen_target_image.to(memory_format=torch.contiguous_format).float()


    return {
        "source_image": source_image, 
        "target_image": target_image,  
        "gen_target_image": gen_target_image,

    }



class RefinedDataset(Dataset):
    def __init__(
        self,
        json_file,
        size=(512,512),   
        s_img_drop_rate=0.0,
        image_root_path="",
        gen_t_img_path =""
    ):
        self.data = json.load(open(json_file))

        self.image_root_path = image_root_path
        self.gen_t_img_path = gen_t_img_path
        self.size = size

        self.s_img_drop_rate = s_img_drop_rate


        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),]
        )




    def __getitem__(self, idx):
        item = self.data[idx]

        s_img_path = os.path.join(self.image_root_path, item["source_image"])
        s_img = Image.open(s_img_path).convert("RGB").resize(self.size, Image.BICUBIC)


        t_img_path = os.path.join(self.image_root_path, item["target_image"])
        t_img = Image.open(t_img_path).convert("RGB").resize(self.size, Image.BICUBIC)

        gen_t_img_path = self.gen_t_img_path + s_img_path.split("/")[-1].replace(".jpg", '_to_') + t_img_path.split("/")[-1].replace(".jpg", '.png')
        gen_t_img = Image.open(gen_t_img_path).convert("RGB").resize(self.size,Image.BICUBIC)


        trans_gen_t_img = self.transform(gen_t_img)
        trans_t_img = self.transform(t_img)

        clip_s_img = (self.clip_image_processor(images=s_img, return_tensors="pt").pixel_values).squeeze(dim=0)



        ## dropout s_img for dinov2
        if random.random() < self.s_img_drop_rate:
            trans_gen_t_img = torch.zeros(trans_gen_t_img.shape)


        return {
            "clip_s_img": clip_s_img,
            "trans_t_img": trans_t_img,
            "trans_gen_t_img": trans_gen_t_img,
        }

    def __len__(self):
        return len(self.data)
