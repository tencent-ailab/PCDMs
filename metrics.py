import os
import pathlib
import torch
import numpy as np
import skimage
from imageio import imread
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import glob
import argparse
import matplotlib.pyplot as plt
from inception import InceptionV3
#from scripts.PerceptualSimilarity.models import dist_model as dm
import lpips
import pandas as pd
import json
import imageio
import cv2
print(skimage.__version__)

class FID():
    """docstring for FID
    Calculates the Frechet Inception Distance (FID) to evalulate GANs
    The FID metric calculates the distance between two distributions of images.
    Typically, we have summary statistics (mean & covariance matrix) of one
    of these distributions, while the 2nd distribution is given by a GAN.
    When run as a stand-alone program, it compares the distribution of
    images that are stored as PNG/JPEG at a specified location with a
    distribution given by summary statistics (in pickle format).
    The FID is calculated by assuming that X_1 and X_2 are the activations of
    the pool_3 layer of the inception net for generated samples and real world
    samples respectivly.
    See --help to see further details.
    Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
    of Tensorflow
    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    def __init__(self):
        self.dims = 2048
        self.batch_size = 128
        self.cuda = True
        self.verbose=False

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx])
        if self.cuda:
            # TODO: put model into specific GPU
            self.model.cuda()

    def __call__(self, images, gt_path):
        """ images:  list of the generated image. The values must lie between 0 and 1.
            gt_path: the path of the ground truth images.  The values must lie between 0 and 1.
        """
        if not os.path.exists(gt_path):
            raise RuntimeError('Invalid path: %s' % gt_path)


        print('calculate gt_path statistics...')
        m1, s1 = self.compute_statistics_of_path(gt_path, self.verbose)
        print('calculate generated_images statistics...')
        m2, s2 = self.calculate_activation_statistics(images, self.verbose)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


    def calculate_from_disk(self, generated_path, gt_path, img_size):
        """ 
        """
        if not os.path.exists(gt_path):
            raise RuntimeError('Invalid path: %s' % gt_path)
        if not os.path.exists(generated_path):
            raise RuntimeError('Invalid path: %s' % generated_path)

        print ('exp-path - '+generated_path)

        print('calculate gt_path statistics...')
        m1, s1 = self.compute_statistics_of_path(gt_path, self.verbose, img_size)
        print('calculate generated_path statistics...')
        m2, s2 = self.compute_statistics_of_path(generated_path, self.verbose, img_size)
        print('calculate frechet distance...')
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        print('fid_distance %f' % (fid_value))
        return fid_value        


    def compute_statistics_of_path(self, path , verbose, img_size):

        size_flag = '{}_{}'.format(img_size[0], img_size[1])
        npz_file = os.path.join(path, size_flag + '_statistics.npz')
        if os.path.exists(npz_file):
            f = np.load(npz_file)
            m, s = f['mu'][:], f['sigma'][:]
            f.close()

        else:

            path = pathlib.Path(path)
            files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

            imgs = (np.array([(cv2.resize(imread(str(fn)).astype(np.float32),img_size,interpolation=cv2.INTER_CUBIC)) for fn in files]))/255.0
            # Bring images to shape (B, 3, H, W)
            imgs = imgs.transpose((0, 3, 1, 2))

            # Rescale images to be between 0 and 1


            m, s = self.calculate_activation_statistics(imgs, verbose)
            np.savez(npz_file, mu=m, sigma=s)

        return m, s  

    def calculate_activation_statistics(self, images, verbose):
        """Calculation of the statistics used by the FID.
        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the
                         number of calculated batches is reported.
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.get_activations(images, verbose)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma            



    def get_activations(self, images, verbose=False):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : the images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size depends
                         on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the number
                         of calculated batches is reported.
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        self.model.eval()

        d0 = images.shape[0]
        if self.batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            self.batch_size = d0

        n_batches = d0 // self.batch_size
        n_used_imgs = n_batches * self.batch_size

        pred_arr = np.empty((n_used_imgs, self.dims))
        for i in range(n_batches):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches))
                      # end='', flush=True)
            start = i * self.batch_size
            end = start + self.batch_size

            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            # batch = Variable(batch, volatile=True)
            if self.cuda:
                batch = batch.cuda()

            pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(self.batch_size, -1)

        if verbose:
            print(' done')

        return pred_arr


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an 
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an 
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


class Reconstruction_Metrics():
    def __init__(self, metric_list=['ssim', 'psnr', 'l1', 'mae'], data_range=1, win_size=51, multichannel=True):
        self.data_range = data_range
        self.win_size = win_size
        self.multichannel = multichannel
        for metric in metric_list:
            if metric in ['ssim', 'psnr', 'l1', 'mae']:
                setattr(self, metric, True)
            else:
                print('unsupport reconstruction metric: %s'%metric)


    def __call__(self, inputs, gts):
        """
        inputs: the generated image, size (b,c,w,h), data range(0, data_range)
        gts:    the ground-truth image, size (b,c,w,h), data range(0, data_range)
        """
        result = dict() 
        [b,n,w,h] = inputs.size()
        inputs = inputs.view(b*n, w, h).detach().cpu().numpy().astype(np.float32).transpose(1,2,0)
        gts = gts.view(b*n, w, h).detach().cpu().numpy().astype(np.float32).transpose(1,2,0)

        if hasattr(self, 'ssim'):
            ssim_value = compare_ssim(inputs, gts, data_range=self.data_range, 
                            win_size=self.win_size, multichannel=self.multichannel) 
            result['ssim'] = ssim_value


        if hasattr(self, 'psnr'):
            psnr_value = compare_psnr(inputs, gts, self.data_range)
            result['psnr'] = psnr_value

        if hasattr(self, 'l1'):
            l1_value = compare_l1(inputs, gts)
            result['l1'] = l1_value            

        if hasattr(self, 'mae'):
            mae_value = compare_mae(inputs, gts)
            result['mae'] = mae_value              
        return result


    def calculate_from_disk(self, inputs, gts,  save_path=None, img_size=(176,256), sort=True, debug=0):
        """
            inputs: .txt files, floders, image files (string), image files (list)
            gts: .txt files, floders, image files (string), image files (list)
        """
        if sort:
            input_image_list = sorted(get_image_list(inputs))
            gt_image_list = sorted(get_image_list(gts))
        else:
            input_image_list = get_image_list(inputs)
            gt_image_list = get_image_list(gts)

        size_flag = '{}_{}'.format(img_size[0], img_size[1])
        npz_file = os.path.join(save_path, size_flag + '_metrics.npz')
        if os.path.exists(npz_file):
            f = np.load(npz_file)
            psnr,ssim,ssim_256,mae,l1=f['psnr'],f['ssim'],f['ssim_256'],f['mae'],f['l1']
        else:
            psnr = []
            ssim = []
            ssim_256 = []
            mae = []
            l1 = []
            names = []

            for index in range(len(input_image_list)):
                name = os.path.basename(input_image_list[index])
                names.append(name)


                img_gt = (cv2.resize(imread(str(gt_image_list[index])).astype(np.float32), img_size,interpolation=cv2.INTER_CUBIC)) /255.0
                img_pred = (cv2.resize(imread(str(input_image_list[index])).astype(np.float32), img_size,interpolation=cv2.INTER_CUBIC)) / 255.0


                if debug != 0:
                    plt.subplot('121')
                    plt.imshow(img_gt)
                    plt.title('Groud truth')
                    plt.subplot('122')
                    plt.imshow(img_pred)
                    plt.title('Output')
                    plt.show()

                psnr.append(compare_psnr(img_gt, img_pred, data_range=self.data_range))
                ssim.append(compare_ssim(img_gt, img_pred, data_range=self.data_range,
                            win_size=self.win_size,multichannel=self.multichannel, channel_axis=2))
                mae.append(compare_mae(img_gt, img_pred))
                l1.append(compare_l1(img_gt, img_pred))

                img_gt_256 = img_gt*255.0
                img_pred_256 = img_pred*255.0
                ssim_256.append(compare_ssim(img_gt_256, img_pred_256, gaussian_weights=True, sigma=1.2,
                                use_sample_covariance=False, multichannel=True, channel_axis=2,
                                data_range=img_pred_256.max() - img_pred_256.min()))

                if np.mod(index, 200) == 0:
                    print(
                        str(index) + ' images processed',
                        "PSNR: %.4f" % round(np.mean(psnr), 4),
                        "SSIM_256: %.4f" % round(np.mean(ssim_256), 4),
                        "MAE: %.4f" % round(np.mean(mae), 4),
                        "l1: %.4f" % round(np.mean(l1), 4),
                    )
            
            if save_path:
                np.savez(save_path + '/' + size_flag + '_metrics.npz', psnr=psnr, ssim=ssim, ssim_256=ssim_256, mae=mae, l1=l1, names=names)

        print(
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "PSNR Variance: %.4f" % round(np.var(psnr), 4),
            "SSIM_256: %.4f" % round(np.mean(ssim_256), 4),
            "SSIM_256 Variance: %.4f" % round(np.var(ssim_256), 4),            
            "MAE: %.4f" % round(np.mean(mae), 4),
            "MAE Variance: %.4f" % round(np.var(mae), 4),
            "l1: %.4f" % round(np.mean(l1), 4),
            "l1 Variance: %.4f" % round(np.var(l1), 4)    
        ) 

        dic = {"psnr":[round(np.mean(psnr), 6)],
               "psnr_variance": [round(np.var(psnr), 6)],
               "ssim_256": [round(np.mean(ssim_256), 6)],
               "ssim_256_variance": [round(np.var(ssim_256), 6)],
               "mae": [round(np.mean(mae), 6)],
               "mae_variance": [round(np.var(mae), 6)],
               "l1": [round(np.mean(l1), 6)],
               "l1_variance": [round(np.var(l1), 6)] } 

        return dic


def get_image_list(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list'%flist)
    return []

def compare_l1(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.mean(np.abs(img_true - img_test))    

def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

def preprocess_path_for_deform_task(gt_path, distorted_path):
    distorted_image_list = sorted(get_image_list(distorted_path))
    gt_list=[]
    distorated_list=[]

    for distorted_image in distorted_image_list:
        image = os.path.basename(distorted_image)[1:]
        image = image.split('_to_')[-1]
        gt_image = gt_path + '/' + image.replace('jpg', 'png')
        if not os.path.isfile(gt_image):
            print(distorted_image, gt_image)
            print('=====')
            continue
        gt_list.append(gt_image)
        distorated_list.append(distorted_image)    

    return gt_list, distorated_list



class LPIPS():
    def __init__(self, use_gpu=True):

        self.model =  lpips.LPIPS(net='alex').eval().cuda()
        self.use_gpu=use_gpu

    def __call__(self, image_1, image_2):
        """
            image_1: images with size (n, 3, w, h) with value [-1, 1]
            image_2: images with size (n, 3, w, h) with value [-1, 1]
        """
        result = self.model.forward(image_1, image_2)
        return result

    def calculate_from_disk(self, path_1, path_2,img_size, batch_size=64, verbose=False, sort=True):

        if sort:
            files_1 = sorted(get_image_list(path_1))
            files_2 = sorted(get_image_list(path_2))
        else:
            files_1 = get_image_list(path_1)
            files_2 = get_image_list(path_2)


        results=[]


        d0 = len(files_1)
        if batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = d0

        n_batches = d0 // batch_size


        for i in range(n_batches):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches))
                      # end='', flush=True)
            start = i * batch_size
            end = start + batch_size

            imgs_1 = np.array([cv2.resize(imread(str(fn)).astype(np.float32),img_size,interpolation=cv2.INTER_CUBIC)/255.0 for fn in files_1[start:end]])
            imgs_2 = np.array([cv2.resize(imread(str(fn)).astype(np.float32),img_size,interpolation=cv2.INTER_CUBIC)/255.0 for fn in files_2[start:end]])

            imgs_1 = imgs_1.transpose((0, 3, 1, 2))
            imgs_2 = imgs_2.transpose((0, 3, 1, 2))

            img_1_batch = torch.from_numpy(imgs_1).type(torch.FloatTensor)
            img_2_batch = torch.from_numpy(imgs_2).type(torch.FloatTensor)

            if self.use_gpu:
                img_1_batch = img_1_batch.cuda()
                img_2_batch = img_2_batch.cuda()

                with torch.no_grad():
                    result = self.model.forward(img_1_batch, img_2_batch)

            results.append(result)


        distance = torch.cat(results,0)[:,0,0,0].mean()

        print('lpips: %.3f'%distance)
        return distance
















