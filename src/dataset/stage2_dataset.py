import os
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from transformers import CLIPImageProcessor

def InpaintCollate_fn(data):


    source_image = torch.stack([example["clip_s_img"] for example in data])
    source_image = source_image.to(memory_format=torch.contiguous_format).float()

 
    target_image = torch.stack([example["clip_t_img"] for example in data])
    target_image = target_image.to(memory_format=torch.contiguous_format).float()



    vae_source_mask_image = torch.stack([example["trans_s_img_mask"] for example in data])
    vae_source_mask_image = vae_source_mask_image.to(memory_format=torch.contiguous_format).float()



    source_target_pose = torch.stack([example["trans_st_pose"] for example in data])
    source_target_pose = source_target_pose.to(memory_format=torch.contiguous_format).float()


    source_target_image = torch.stack([example["trans_st_img"] for example in data])
    source_target_image = source_target_image.to(memory_format=torch.contiguous_format).float()



    return {

        "source_image": source_image, 
        "target_image": target_image, 
        "vae_source_mask_image": vae_source_mask_image,
        "source_target_pose": source_target_pose,
        "source_target_image": source_target_image,
    }



class InpaintDataset(Dataset):
    def __init__(
        self,
        json_file,
        image_root_path,
        size=(512,512),   
        imgp_drop_rate=0.0,
        imgg_drop_rate=0.0,
    ):
        if isinstance(json_file, str):
            self.data = json.load(open(json_file))
            print(len(self.data))
        else:
            self.data = json_file
        self.image_root_path = image_root_path


        self.size = size

        self.imgp_drop_rate = imgp_drop_rate
        self.imgg_drop_rate = imgg_drop_rate

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),]
        )




    def __getitem__(self, idx):
        item = self.data[idx]

        s_img_path = os.path.join(self.image_root_path, item["source_image"].replace(".jpg", ".png"))
        s_img = Image.open(s_img_path).convert("RGB").resize(self.size, Image.BICUBIC)

        black_img = Image.new("RGB", self.size, (0, 0, 0))
        s_img_mask = Image.new("RGB", (self.size[0] * 2, self.size[1]))
        s_img_mask.paste(s_img, (0, 0))
        s_img_mask.paste(black_img, (self.size[0], 0))

        t_img_path = os.path.join(self.image_root_path, item["target_image"].replace(".jpg", ".png"))
        t_img = Image.open(t_img_path).convert("RGB").resize(self.size, Image.BICUBIC)

        st_img = (Image.new("RGB", (self.size[0] * 2, self.size[1])))
        st_img.paste(s_img, (0, 0))
        st_img.paste(t_img, (self.size[0], 0))

        s_pose = Image.open(s_img_path.replace("/train_all_png/", "/openpose_all_img/").replace(".png", "_pose.jpg")).convert("RGB").resize(self.size, Image.BICUBIC)
        t_pose = Image.open(t_img_path.replace("/train_all_png/", "/openpose_all_img/").replace(".png", "_pose.jpg")).convert("RGB").resize(self.size, Image.BICUBIC)
        st_pose = Image.new("RGB", (self.size[0] * 2, self.size[1]))
        st_pose.paste(s_pose, (0, 0))
        st_pose.paste(t_pose, (self.size[0], 0))


        trans_s_img_mask = self.transform(s_img_mask)
        trans_st_img = self.transform(st_img)
        trans_st_pose = self.transform(st_pose)



        clip_s_img = (self.clip_image_processor(images=s_img, return_tensors="pt").pixel_values).squeeze(dim=0)
        clip_t_img = (self.clip_image_processor(images=t_img, return_tensors="pt").pixel_values).squeeze(dim=0)


        ## dropout s_img for dinov2
        if random.random() < self.imgp_drop_rate:
            clip_s_img = torch.zeros(clip_s_img.shape)

        ## dropout t_img_embed
        if random.random() < self.imgg_drop_rate:
            clip_t_img = torch.zeros(clip_t_img.shape)

        return {
            "clip_s_img": clip_s_img,
            "clip_t_img": clip_t_img,
            "trans_st_img": trans_st_img,
            "trans_st_pose": trans_st_pose,
            "trans_s_img_mask": trans_s_img_mask,
        }

    def __len__(self):
        return len(self.data)
