from metrics import FID, LPIPS, Reconstruction_Metrics, preprocess_path_for_deform_task
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid = FID()
lpips_obj = LPIPS()
rec = Reconstruction_Metrics()

real_path = './datasets/deepfashing/train_lst_256_png'
gt_path = '/datasets/deepfashing/test_lst_256_png'


distorated_path = './PCDMs_Results/stage3_256_results'   
results_save_path =  distorated_path + '_results.txt'    # save path


gt_list, distorated_list = preprocess_path_for_deform_task(gt_path, distorated_path)
print(len(gt_list), len(distorated_list))

FID = fid.calculate_from_disk(distorated_path, real_path, img_size=(176,256))
LPIPS = lpips_obj.calculate_from_disk(distorated_list, gt_list, img_size=(176,256), sort=False)
REC = rec.calculate_from_disk(distorated_list, gt_list, distorated_path,  img_size=(176,256), sort=False, debug=False)

print ("FID: "+str(FID)+"\nLPIPS: "+str(LPIPS)+"\nSSIM: "+str(REC))
with open(results_save_path, 'a') as ff:
    ff.write("\nFID: "+str(FID)+"\nLPIPS: "+str(LPIPS)+"\nSSIM: "+str(REC))