# PCDMs
Implementation code：[Advancing Pose-Guided Image Synthesis with Progressive Conditional Diffusion Models](https://arxiv.org/pdf/2310.06313.pdf), published at International Conference on Learning Representations (ICLR) 2024.


## Generated Results
![PCDMs Motivation](imgs/compare_sota.png)

![PCDMs results1](imgs/demo.png)

![PCDMs results2](imgs/demo2.png)

You can directly download our test results from Google Drive: (1) [PCDMs vs SOTA](https://drive.google.com/drive/folders/1q21tA3VsQqScecQ7m3_eUFxIPWUGYKAa?usp=drive_link) (2) [PCDMs Results](https://drive.google.com/drive/folders/1sjqMhZ79pugk2IHhW-whg_NASpx3BSew?usp=drive_link).

The [PCDMs vs SOTA](https://drive.google.com/drive/folders/1q21tA3VsQqScecQ7m3_eUFxIPWUGYKAa?usp=drive_link) compares our method with several state-of-the-art methods e.g. ADGAN, PISE, GFLA, DPTN, CASD, NTED, PIDM. 
Each row contains target_pose, source_image, ground_truth, ADGAN, PISE, GFLA, DPTN, CASD, NTED, PIDM, and PCDMs (ours) respectively.

The weights can be obtained from [Google drive](https://drive.google.com/file/d/1H7WafGJrJZiblClJp0-AwcZrk1aNokWP/view?usp=drive_link).

## Installation
```
# install diffusers & pose extractor
pip install diffusers==0.24.0
pip install controlnet-aux==0.0.7


# install DWPose which is dependent on MMDetection, MMCV and MMPose
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

# clone code
git clone https://github.com/tencent-ailab/PCDMs.git

# download the models
cd PCDMs
mv {weights} ./PCDMs_ckpt.pt

# then you can use the notebook
```


## Methods

![PCDMs Motivation](imgs/diagram.png)


![PCDMs Framework](imgs/frameworkv2.png)

## Dataset
### Processed Data
This [link](https://drive.google.com/drive/folders/17yqf2iV2y7Hwx-__uV_-cRaOm2XX8lWS?usp=drive_link) contains processed and prepared data that is ready for use.
The data has been processed in the following ways:

&#8226; Rename image

&#8226; Split the train/test set

&#8226; keypoints extracted with Openpose

The folder structure of dataset should be as follows:

```
Deepfashion/
├── all_data_png                        # including train and test images
│   ├── img1.png          
│   ├── ...
│   ├── img52712.png         
├── train_lst_256_png                   # including train images of 256 size
│   ├── img1.png
│   ├── ...
│   ├── img48674.png
├── train_lst_512_png                   # including train images of 512 size
│   ├── img1.png
│   ├── ...
│   ├── img48674.png
├── test_lst_256_png                    # including test images of 256 size
│   ├── img1.png
│   ├── ...
│   ├── img4038.png
├── test_lst_512_png                    # including test images of 512 size
│   ├── img1.png
│   ├── ...
│   ├── img4038.png
├── normalized_pose_txt.zip             # including pose coordinate of train and test set
│   ├── pose_coordinate1.txt
│   ├── ...
│   ├── pose_coordinate40160.txt
├── train_data.json                     
├── test_data.json
```

### Original Data
 Download ```img_highres.zip``` of the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00?resourcekey=0-fsjVShvqXP2517KnwaZ0zw). 
 
 Unzip ```img_highres.zip```. You will need to ask for password from the dataset maintainers. 


## Checkpoints Links
We provide 3 stage checkpoints available [here](https://drive.google.com/drive/folders/1xDpSIBdP11UoXe0HoWGPWBfEnk1zRMVw?usp=drive_link).



## How to Train/Test
1. train/test stage1-prior
```
sh run_stage1.sh  & sh run_test_stage1.sh
```

2. train/test stage2-inpaint
```
sh run_stage2.sh  & sh run_test_stage2.sh
```

3. train/test stage3-refined
```
sh run_stage3.sh  & sh run_test_stage3.sh
```

## Citation
If this work is useful to you, please consider citing our paper:
```
@article{shen2023advancing,
  title={Advancing Pose-Guided Image Synthesis with Progressive Conditional Diffusion Models},
  author={Shen, Fei and Ye, Hu and Zhang, Jun and Wang, Cong and Han, Xiao and Yang, Wei},
  journal={arXiv preprint arXiv:2310.06313},
  year={2023}
}

```

