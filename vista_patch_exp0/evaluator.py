import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
from scipy.stats import mode


import tensorflow as tf
device_name = tf.test.gpu_device_name()


import tensorflow as tf
device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))


import segmentation_models_3D as sm
from skimage import io
import numpy as np
import tifffile


import argparse
parser = argparse.ArgumentParser(description='Enter crop type numbers in order')
parser.add_argument('--season', type=str, default=0, help='input season')
parser.add_argument('--num_subgroup_samples', type=int, default=300, help='number of samples per subgroup')


args = parser.parse_args()
season = args.season
num_subgroup_samples = args.num_subgroup_samples  # this by default can assume a maximum value of 1000 . It can be further increased by editing the test_set_sampler


vista_crop_dict = {0:'NA', 10:'NA' , 11: 'ALFALFA', 12: 'BEET', 13: 'CLOVER', 14: 'FLAX', 15: 'FLOWERING_LEGUMES', 16: 'FLOWERS', 17: 'FOREST', 18: 'GRAIN_MAIZE', 19: 'GRASSLAND', 20: 'HOPS', 21: 'LEGUMES', 22: 'VISTA_NA', 23: 'PERMANENT_PLANTATIONS', 24: 'PLASTIC', 25: 'POTATO', 26: 'PUMPKIN', 27: 'RICE', 28: 'SILAGE_MAIZE', 29: 'SOY', 30: 'SPRING_BARLEY', 31: 'SPRING_OAT', 32: 'SPRING_OTHER_CEREALS', 33: 'SPRING_RAPESEED', 34: 'SPRING_RYE', 35: 'SPRING_SORGHUM', 36: 'SPRING_SPELT', 37: 'SPRING_TRITICALE', 38: 'SPRING_WHEAT', 39: 'SUGARBEET', 40: 'SUNFLOWER', 41: 'SWEET_POTATOES', 42: 'TEMPORARY_GRASSLAND', 43: 'WINTER_BARLEY', 44: 'WINTER_OAT', 45: 'WINTER_OTHER_CEREALS', 46: 'WINTER_RAPESEED', 47: 'WINTER_RYE', 48: 'WINTER_SORGHUM', 49: 'WINTER_SPELT', 50: 'WINTER_TRITICALE', 51: 'WINTER_WHEAT'}

import gc
BACKBONE = 'vgg16'  

if season == "Feb_Aug":
    chosen_season_crops = [10, 43, 44, 46, 47, 50, 51]
    chosen_crop_types_list_list = [[43, 46, 51], [44, 47, 50]] 
    chosen_crop_types_list_list_models = [[43, 46, 51], [44, 47, 50]]

if season == "May_Aug":
    chosen_season_crops = [10, 12, 25, 30, 31, 33, 38]
    chosen_crop_types_list_list = [[12, 25, 30], [31, 33, 38]] 
    chosen_crop_types_list_list_models = [[12, 25, 30], [31, 33, 38]]

if season == "Jun_Oct":
    chosen_season_crops = [10, 18, 19, 40, 17, 28, 29]
    chosen_crop_types_list_list = [[18, 19, 40], [17, 28, 29]] 
    chosen_crop_types_list_list_models = [[18, 19, 40], [17, 28, 29]]

if season == "Jan_Aug":
    chosen_season_crops = [10, 14, 17, 19]
    chosen_crop_types_list_list = [[14, 17, 19]] 
    chosen_crop_types_list_list_models = [[14, 17, 19]]


crop_types_all_list = [ 10, 11,  12,  13,  14,  15,  17,  18,  19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 37, 38, 40, 42, 43, 44, 45, 46, 47, 50, 51]

for test_img_number in range(num_subgroup_samples):
    for chosen_subset_for_test_set in range(len(chosen_crop_types_list_list)):

        stacked_test_preds = []
        num_epochs = 120

        X_test = io.imread('./storage/test_sets_of_subsets_all/X_test_contains'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][0]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][1]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][2]]+'_.tif')
        y_test_1 = io.imread('./storage/test_sets_of_subsets_all/y_test_all_contains'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][0]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][1]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][2]]+'_.tif')

        ground_truth_1 = y_test_1[test_img_number-1] + 10

        for k in crop_types_all_list:
            if not(k in chosen_season_crops):
                ground_truth_1[ground_truth_1==k]=0


        ensambled_ground_truth = np.zeros((64, 64))
        ensambled_result_image = np.zeros((64, 64))
        ensambled_result_image_gen = np.zeros((64, 64, 64))

        test_img = X_test[test_img_number-1]
        tifffile.imwrite('./ensamble_results/iou_f1_class_weights/inputs/input_subset_season'+season+'_'+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.tif', test_img)

        
        test_img_input1=np.expand_dims(test_img, 0)
        
        ground_truth_flattened = ground_truth_1[0,:,:]

        ll = 0 
        for chosen_crop_types_list in chosen_crop_types_list_list_models:

            gc.collect()
            my_model_1 = sm.Unet(BACKBONE, input_shape=(64, 64, 64, 3), classes=4, encoder_weights=None, activation='softmax')
            my_model_1.load_weights('./checkpoints_f1/3D_unet_g4_h4_crop_'+str(chosen_crop_types_list[0]-10)+'_'+str(chosen_crop_types_list[1]-10)+'_'+str(chosen_crop_types_list[2]-10)+'_epoch_f.h5', by_name=True, skip_mismatch=True)

            test_pred1 = my_model_1.predict(test_img_input1)
            del my_model_1
            gc.collect()
            test_prediction1 = np.argmax(test_pred1, axis=4)[0,:,:,:]

            ll+=1
            for i in range(3):
                test_prediction1[test_prediction1==i+1]= chosen_crop_types_list[i]
            for i in range(3):
                ensambled_result_image_gen[test_prediction1==chosen_crop_types_list[i]]=chosen_crop_types_list[i]
            stacked_test_preds.append(ensambled_result_image_gen.copy())
            ensambled_result_image_gen = np.zeros((64, 64, 64))

        stacked_test_preds = np.concatenate((stacked_test_preds), axis=0)
        ensambled_result_image_gen = np.median(stacked_test_preds, axis=0).astype(np.uint8)

        test_mode = np.zeros((64, 64))
        for i in range(64):
            for j in range(64):
                see = stacked_test_preds[:, i, j]
                sum_see = np.sum(see)
                if sum_see == 0:
                    test_mode[i, j] = 0
                else:
                    local_mode = mode(see[see>0])[0]
                    test_mode[i, j]= local_mode
                    
        test_mode = test_mode.astype(np.uint8)
        ground_truth_flattened = ground_truth_flattened.astype(np.uint8)

        tifffile.imwrite('./ensamble_results/iou_f1_class_weights/no_cloud_interpol/ground_truth_subset_season'+season+'_'+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.tif', test_mode)
        tifffile.imwrite('./ensamble_results/iou_f1_class_weights/no_cloud_interpol/prediction_subset_season'+season+'_'+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.tif', ground_truth_flattened)


