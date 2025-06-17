import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
from keras.models import load_model
from scipy.stats import mode


import tensorflow as tf
device_name = tf.test.gpu_device_name()


import tensorflow as tf
'''device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')'''
print('Found GPU at: {}'.format(device_name))


import segmentation_models_3D as sm
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tifffile
import glob
import gc

import rasterio
from rasterio.transform import from_origin


#preprocess_input = sm.get_preprocessing(BACKBONE)

'''


export CUDA_VISIBLE_DEVICES=0
conda deactivate
conda deactivate
conda activate /home/luser/miniforge3/envs/stcon4
cd stelar_3dunet/
python3 vista_patch_exp0/vista_testing_comp_f1.py --season Feb_Aug


export CUDA_VISIBLE_DEVICES=1
conda deactivate
conda deactivate
conda activate /home/luser/miniforge3/envs/stcon4
cd stelar_3dunet/
python3 vista_patch_exp0/vista_testing_comp_f1.py --season May_Aug


export CUDA_VISIBLE_DEVICES=1
conda deactivate
conda deactivate
conda activate /home/luser/miniforge3/envs/stcon4
cd stelar_3dunet/
python3 vista_patch_exp0/vista_testing_comp_f1.py --season Jun_Oct

export CUDA_VISIBLE_DEVICES=1
conda deactivate
conda deactivate
conda activate /home/luser/miniforge3/envs/stcon4
cd stelar_3dunet/
python3 vista_patch_exp0/vista_testing_comp_f1.py --season Jan_Aug

'''
'''


if season=="Feb_Aug":
    time_strip_inds = [0, 40, 80, 120, 150, 175]  # for Feb_Aug 
if season=="May_Aug":
    time_strip_inds = [20, 30, 120, 130, 175, 180] # for May_Aug
if season=="Jun_Oct":
    time_strip_inds = [40, 50, 130, 140, 180, 190] # for Jun_Oct
if season=="Jan_Aug":
    time_strip_inds = [0, 30, 70, 110, 150, 180] # for Jan_Aug

'''


import argparse
parser = argparse.ArgumentParser(description='Enter crop type numbers in order')
from skimage.util import view_as_windows

#parser.add_argument('--g', type=int, default=0, help='Select crop type')
#parser.add_argument('--h', type=int, default=0, help='Select crop type')
parser.add_argument('--season', type=str, default=0, help='input season')

args = parser.parse_args()

#g = args.g
#h = args.h
season = args.season

class_weights =  True
cloud_interpolation = False

prefix = '/app/'



vista_crop_dict = {0:'NA', 10:'NA' , 11: 'ALFALFA', 12: 'BEET', 13: 'CLOVER', 14: 'FLAX', 15: 'FLOWERING_LEGUMES', 16: 'FLOWERS', 17: 'FOREST', 18: 'GRAIN_MAIZE', 19: 'GRASSLAND', 20: 'HOPS', 21: 'LEGUMES', 22: 'VISTA_NA', 23: 'PERMANENT_PLANTATIONS', 24: 'PLASTIC', 25: 'POTATO', 26: 'PUMPKIN', 27: 'RICE', 28: 'SILAGE_MAIZE', 29: 'SOY', 30: 'SPRING_BARLEY', 31: 'SPRING_OAT', 32: 'SPRING_OTHER_CEREALS', 33: 'SPRING_RAPESEED', 34: 'SPRING_RYE', 35: 'SPRING_SORGHUM', 36: 'SPRING_SPELT', 37: 'SPRING_TRITICALE', 38: 'SPRING_WHEAT', 39: 'SUGARBEET', 40: 'SUNFLOWER', 41: 'SWEET_POTATOES', 42: 'TEMPORARY_GRASSLAND', 43: 'WINTER_BARLEY', 44: 'WINTER_OAT', 45: 'WINTER_OTHER_CEREALS', 46: 'WINTER_RAPESEED', 47: 'WINTER_RYE', 48: 'WINTER_SORGHUM', 49: 'WINTER_SPELT', 50: 'WINTER_TRITICALE', 51: 'WINTER_WHEAT'}
labels = np.load(prefix+'/storage/full_mast/vista_labes_aligned.npy').astype(np.uint8)

print("labels.shape", labels.shape)

#season = "winter"
#season = "spring"

winter_crop_types = [10, 43, 44, 46, 47, 50, 51]
spring_crop_types = [10, 12, 25, 30, 31, 33, 38]
summer_autumn_crop_types = [10, 18, 19, 40, 17, 28, 29]
#summer_autumn_crop_types = [10, 17, 28, 29]
winter_spring_summer_crop_types = [10, 14, 17, 19]


if season == "May_Aug":
    chosen_season_crops = spring_crop_types
    time_strip_ind = 175

if season == "Feb_Aug":
    chosen_season_crops = winter_crop_types
    time_strip_ind = 150

if season == "Jun_Oct":
    chosen_season_crops = summer_autumn_crop_types
    time_strip_ind = 180

if season == "Jan_Aug":
    chosen_season_crops = winter_spring_summer_crop_types
    time_strip_ind = 140

#g = 0
#h = 0
#winter_crop_types = [0, 33, 34, 35, 36, 37, 40, 41]


def slice_and_stack_cubes(data, cube_size=64):
    z_dim, x_dim, y_dim = data.shape
    cubes = []

    # Iterate through the spatial dimensions in steps of cube_size
    for x in range(0, x_dim, cube_size):
        for y in range(0, y_dim, cube_size):
            # Check if the sub-cube fits within the boundaries
            if x + cube_size <= x_dim and y + cube_size <= y_dim:
                cube = data[:, x:x + cube_size, y:y + cube_size]
                cubes.append(cube)

    # Stack the collected cubes into a new array
    cubes_array = np.stack(cubes)
    return cubes_array

def reassemble_cubes(cubes, original_shape, cube_size=64):
    z_dim, x_dim, y_dim = original_shape
    reassembled = np.zeros((z_dim, x_dim, y_dim))

    index = 0
    for x in range(0, x_dim, cube_size):
        for y in range(0, y_dim, cube_size):
            if x + cube_size <= x_dim and y + cube_size <= y_dim:
                reassembled[:, x:x + cube_size, y:y + cube_size] = cubes[index]
                index += 1

    return reassembled

def reassemble_2d_slices(slices, original_shape, slice_size=64):
    x_dim, y_dim = original_shape
    reassembled = np.zeros((x_dim, y_dim))

    index = 0
    for x in range(0, x_dim, slice_size):
        for y in range(0, y_dim, slice_size):
            if x + slice_size <= x_dim and y + slice_size <= y_dim:
                reassembled[x:x + slice_size, y:y + slice_size] = slices[index]
                index += 1

    return reassembled


def slice_2d_labels(labels, slice_size=64):
    x_dim, y_dim = labels.shape
    slices = []

    # Iterate through the spatial dimensions in steps of slice_size
    for x in range(0, x_dim, slice_size):
        for y in range(0, y_dim, slice_size):
            # Check if the sub-slice fits within the boundaries
            if x + slice_size <= x_dim and y + slice_size <= y_dim:
                slice = labels[x:x + slice_size, y:y + slice_size]
                slices.append(slice)

    # Stack the collected slices into a new array
    slices_array = np.stack(slices)
    return slices_array


def get_labels_in_color(groud_truth_image):
    color_map = {10: [0, 0, 0], 0: [0, 0, 0], 11: [0, 255, 0], 12: [0, 0, 255], 13: [255, 255, 0], 14: [255, 165, 0], 15: [255, 0, 255], 16: [0, 255, 255], 17: [128, 0, 128], 18: [128, 128, 0], 19: [0, 128, 0], 20: [128, 0, 0], 21: [0, 0, 128], 22: [128, 128, 128], 23: [0, 128, 128], 24: [255, 0, 0], 25: [255, 255, 255], 26: [192, 192, 192], 27: [139, 0, 0], 28: [0, 100, 0], 29: [0, 0, 139], 30: [255, 215, 0], 31: [255, 140, 0], 32: [139, 0, 139], 33: [0, 206, 209], 34: [75, 0, 130], 35: [85, 107, 47], 36: [34, 139, 34], 37: [165, 42, 42], 38: [70, 130, 180], 39: [169, 169, 169], 40: [32, 178, 170], 41: [47, 79, 79], 42: [245, 245, 245], 43: [105, 105, 105], 44: [205, 92, 92], 45: [50, 205, 50], 46: [65, 105, 225], 47: [255, 223, 0], 48: [255, 99, 71], 49: [186, 85, 211], 50: [0, 191, 255], 51: [192, 192, 192]}

    groud_truth_color_image = np.zeros(groud_truth_image.shape + (3,), dtype=np.uint8)
    for i in range(groud_truth_image.shape[0]):
        for j in range(groud_truth_image.shape[1]):
            segment_id_gt = groud_truth_image[i, j]
            groud_truth_color_image[i, j] = color_map[segment_id_gt]
    return groud_truth_color_image


def replace_zeros_with_average(arr):
    for i in range(len(arr)):
        if arr[i] == 0:
            left = i - 1
            right = i + 1
            while left >= 0 and arr[left] == 0:
                left -= 1
            while right < len(arr) and arr[right] == 0:
                right += 1
            if left >= 0 and right < len(arr):
                arr[i] = (arr[left] + arr[right]) // 2
            elif left >= 0:
                arr[i] = arr[left]
            elif right < len(arr):
                arr[i] = arr[right]
    return arr

color_map = {10: [0, 0, 0], 0: [0, 0, 0], 11: [0, 255, 0], 12: [0, 0, 255], 13: [255, 255, 0], 14: [255, 165, 0], 15: [255, 0, 255], 16: [0, 255, 255], 17: [128, 0, 128], 18: [128, 128, 0], 19: [0, 128, 0], 20: [128, 0, 0], 21: [0, 0, 128], 22: [128, 128, 128], 23: [0, 128, 128], 24: [255, 0, 0], 25: [255, 255, 255], 26: [192, 192, 192], 27: [139, 0, 0], 28: [0, 100, 0], 29: [0, 0, 139], 30: [255, 215, 0], 31: [255, 140, 0], 32: [139, 0, 139], 33: [0, 206, 209], 34: [75, 0, 130], 35: [85, 107, 47], 36: [34, 139, 34], 37: [165, 42, 42], 38: [70, 130, 180], 39: [169, 169, 169], 40: [32, 178, 170], 41: [47, 79, 79], 42: [245, 245, 245], 43: [105, 105, 105], 44: [205, 92, 92], 45: [50, 205, 50], 46: [65, 105, 225], 47: [255, 223, 0], 48: [255, 99, 71], 49: [186, 85, 211], 50: [0, 191, 255], 51: [192, 192, 192]}


import gc
BACKBONE = 'vgg16'  
preprocess_input = sm.get_preprocessing(BACKBONE)




if class_weights:

    if(season=="Feb_Aug"):
        chosen_crop_types_list_list = [[43, 46, 51], [44, 47, 50]] # winter crop inds
        chosen_crop_types_list_list_models = [[43, 46, 51], [44, 47, 50]]


    if(season=="May_Aug"):
        chosen_crop_types_list_list = [[12, 25, 30], [31, 33, 38]] # spring crop inds
        chosen_crop_types_list_list_models = [[12, 25, 30], [31, 33, 38]]

    if(season=="Jun_Oct"):
        chosen_crop_types_list_list = [[18, 19, 40], [17, 28, 29]] # summer_autumn crop inds
        chosen_crop_types_list_list_models = [[18, 19, 40], [17, 28, 29]]

        #chosen_crop_types_list_list = [[17, 28, 29]] # summer_autumn crop inds
        #chosen_crop_types_list_list_models = [[17, 28, 29]]

    if(season=="Jan_Aug"):
        chosen_crop_types_list_list = [[14, 17, 19]] # winter_spring_summer crop inds
        chosen_crop_types_list_list_models = [[14, 17, 19]]


else:

    if(season=="Feb_Aug"):
        chosen_crop_types_list_list = [[43, 46, 51], [44, 47, 50]] # winter crop inds
        chosen_crop_types_list_list_models = [[43, 46, 51], [44, 47, 50]]

    if(season=="May_Aug"):
        chosen_crop_types_list_list = [[12, 25, 30], [31, 33, 38]] # spring crop inds
        chosen_crop_types_list_list_models = [[12, 25, 30], [31, 33, 38]]

    if(season=="Jun_Oct"):
        chosen_crop_types_list_list = [[18, 19, 40], [17, 28, 29]] # summer_autumn crop inds
        chosen_crop_types_list_list_models = [[18, 19, 40], [17, 28, 29]]

        #chosen_crop_types_list_list = [[17, 28, 29]] # summer_autumn crop inds
        #chosen_crop_types_list_list_models = [[17, 28, 29]]

    if(season=="Jan_Aug"):
        chosen_crop_types_list_list = [[14, 17, 19]] # winter_spring_summer crop inds
        chosen_crop_types_list_list_models = [[14, 17, 19]]


crop_types_all_list = [ 10, 11,  12,  13,  14,  15,  17,  18,  19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 37, 38, 40, 42, 43, 44, 45, 46, 47, 50, 51]

#chosen_crop_types_list = winter_crop_types




#test_img_number = 159
#chosen_subset_for_test_set = 4



###########################
from segmentation_models_3D import get_preprocessing
preprocess_input = get_preprocessing('vgg16')
##########################

filepaths = glob.glob(prefix+'/dataset/france2/processed_lai_npy/*.npy')
filepaths.sort()


considered_filepaths = filepaths[time_strip_ind:time_strip_ind+64]#+filepaths[time_strip_ind:time_strip_ind+20]+filepaths[time_strip_ind:time_strip_ind+20]+filepaths[time_strip_ind:time_strip_ind+20]

print("len(considered_filepaths)", len(considered_filepaths))

print("considered_filepaths", considered_filepaths)

numpy_array_all = []
pad_height = 46
pad_width = 46
k = 0
for filepath in considered_filepaths:
    numpy_array = np.load(filepath).astype(np.float32)
    tf_tensor = tf.convert_to_tensor(numpy_array)  # defaults to float32 already
    tf_tensor_padded = np.pad(tf_tensor, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    tf_tensor_padded = tf.identity(tf_tensor_padded)  # placeholder; TF automatically uses GPU if available

    print("tf_tensor_padded.shape", tf_tensor_padded.shape)

    '''print("tf_tensor.shape", tf_tensor.shape)
    plt.imshow(tf_tensor)
    plt.savefig('/data1/chethan/stelar_3dunet/vista_patch_exp0/each/fig'+str(k)+'.png')
    plt.close()'''
    k+=1
    numpy_array_all.append(tf_tensor_padded)

# Stack all tensors along axis 0
all_processed_LAI = np.stack(numpy_array_all, axis=0)
print("all_processed_LAI.shape", all_processed_LAI.shape)
del numpy_array_all
gc.collect()


labels = tf.convert_to_tensor(labels)  # defaults to float32 already
labels = np.pad(labels, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
labels = tf.identity(labels)  # placeholder; TF automatically uses GPU if available


all_processed_LAI = tf.convert_to_tensor(all_processed_LAI, dtype=tf.float32)

print("all_processed_LAI.device", all_processed_LAI.device)


all_processed_LAI_stacked = []
all_labels = []
for i in range(all_processed_LAI.shape[-1]//64):
    for j in range(all_processed_LAI.shape[-2]//64):
        print("i", i)
        print("j", j)
        print("all_processed_LAI[:, 0:64, 0:64].shape", all_processed_LAI[:, i*64:(i*64)+64, j*64:(j*64)+64].shape)

        '''sel = all_processed_LAI[:, i*64:(i*64)+64, j*64:(j*64)+64]
        print("sel.shape", sel.shape)
        plt.figure(figsize=(10,10))
        plt.title(' Time series LAI input')
        for m in range(64):
            plt.axis('off')
            plt.subplot(8, 8, m+1)
            plt.imshow(sel[m,:,:])
            plt.axis('off')
            plt.savefig('/data1/chethan/stelar_3dunet/vista_patch_exp0/demos/ch/'+season+'LAI_subset_'+str(i)+'_sample_'+str(j)+'_.png', bbox_inches='tight')
        plt.close()'''

        all_processed_LAI_stacked.append(all_processed_LAI[:, i*64:(i*64)+64, j*64:(j*64)+64])

        all_labels.append(labels[i*64:(i*64)+64, j*64:(j*64)+64])

del all_processed_LAI
gc.collect()

#sliced_cubes = tf.stack(all_processed_LAI_stacked)

sliced_cubes = np.stack(all_processed_LAI_stacked, axis=0)
sliced_cubes = tf.convert_to_tensor(sliced_cubes, dtype=tf.float32)


sliced_labels = np.stack(all_labels, axis=0)
sliced_labels = tf.convert_to_tensor(sliced_labels, dtype=tf.float32)

del all_labels
del labels
gc.collect()

#print("stack_LAI.shape", stack_LAI.shape)


# Execute the function
#sliced_cubes = slice_and_stack_cubes(all_processed_LAI)

# Output the shape of the resulting array
print("sliced_cubes.shape", sliced_cubes.shape)


#sliced_labels = slice_2d_labels(labels)

#sliced_cubes = sliced_cubes.astype(np.uint8)

print("sliced_labels.shape", sliced_labels.shape)

X_test = np.stack((sliced_cubes,)*3, axis=-1)
sliced_labels = np.repeat(sliced_labels[:, np.newaxis, :, :], repeats=64, axis=1)
y_test_1 = sliced_labels

del sliced_cubes
del sliced_labels
gc.collect()

print("y_test_1.shape", y_test_1.shape)
print("X_test.shape", X_test.shape)


#ground_truth_bl = np.zeros((10048, 10048, 3))
#predicted_bl = np.zeros((10048, 10048, 3))

check = 0
all_test_mode = []
all_ground_truth_flattened = []
for test_img_number in range(len(X_test)):
    #for chosen_subset_for_test_set in range(len(chosen_crop_types_list_list)):
    #for chosen_subset_for_test_set in range(1):
    chosen_subset_for_test_set = 0
    stacked_test_preds = []
    num_epochs = 120

    #X_test = io.imread('./storage/test_sets_of_subsets_all/X_test_contains'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][0]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][1]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][2]]+'_.tif')
    #y_test_1 = io.imread('./storage/test_sets_of_subsets_all/y_test_all_contains'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][0]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][1]]+'_'+vista_crop_dict[chosen_crop_types_list_list[chosen_subset_for_test_set][2]]+'_.tif')

    #print("y_test_1.shape", y_test_1.shape)
    #print("X_test.shape", X_test.shape)


    '''X_test = np.stack((sliced_cubes,)*3, axis=-1)
    sliced_labels = np.repeat(sliced_labels[:, np.newaxis, :, :], repeats=64, axis=1)
    y_test_1 = sliced_labels

    print("y_test_1.shape", y_test_1.shape)
    print("X_test.shape", X_test.shape)'''

    ground_truth_1 = y_test_1[test_img_number-1] + 10

    for k in crop_types_all_list:
        if not(k in chosen_season_crops):
            ground_truth_1[ground_truth_1==k]=0


    ensambled_ground_truth = np.zeros((64, 64))
    ensambled_result_image = np.zeros((64, 64))
    ensambled_result_image_gen = np.zeros((64, 64, 64))

    test_img = X_test[test_img_number-1]
    #tifffile.imsave('/data1/chethan/stelar_3dunet/vista_patch_exp0/ensamble_results/iou_f1_class_weights/inputs/input_subset_season'+season+'_'+str(chosen_subset_for_test_set)+'_sample_'+str(test_img_number)+'_.tif', test_img)

    if cloud_interpolation:
        for i in range(64):
            for j in range(64):
                for c in range(3):
                    exp_time_strip = test_img[:, i, j, c]
                    exp_time_strip_av = replace_zeros_with_average(exp_time_strip)
                    test_img[:, i, j, c] = exp_time_strip_av
    #test_img = test_img.astype(np.uint8)            ###########################  was this a trouble maker
    test_img_input=np.expand_dims(test_img, 0)
    #test_img_input1 = preprocess_input(test_img_input, backend='tf')
    #test_img_input1 = preprocess_input(test_img_input)     ### works for winter_spring_summer, summer_autumn
    
    test_img_input1 = test_img_input # no preprocessing for spring and winter

    '''if season == "spring":
        test_img_input1 = test_img_input
    if season == "winter":
        test_img_input1 = test_img_input
    if season == "winter_spring_summer":
        test_img_input1 = test_img_input #preprocess_input(test_img_input) 
    if season == "summer_autumn":
        test_img_input1 = test_img_input''' #preprocess_input(test_img_input) 
        

    ground_truth_flattened = ground_truth_1[0,:,:]

    ll = 0 
    for chosen_crop_types_list in chosen_crop_types_list_list_models:

        gc.collect()
        #if class_weights:
        #my_model_1 = load_model('./checkpoints/3D_unet_g3_h3_crop_'+str(chosen_crop_types_list[0]-10)+'_'+str(chosen_crop_types_list[1]-10)+'_'+str(chosen_crop_types_list[2]-10)+'_epoch_'+str(num_epochs)+'.h5', compile=False)

        BACKBONE = 'vgg16'
        my_model_1 = sm.Unet(BACKBONE, input_shape=(64, 64, 64, 3), classes=4, encoder_weights=None, activation='softmax')
        #print("chosen_crop_types_list", chosen_crop_types_list)
        my_model_1.load_weights(prefix+'/checkpoints_f1/3D_unet_g4_h4_crop_'+str(chosen_crop_types_list[0]-10)+'_'+str(chosen_crop_types_list[1]-10)+'_'+str(chosen_crop_types_list[2]-10)+'_epoch_f.h5', by_name=True, skip_mismatch=True)


        #'/data1/chethan/stelar_3dunet/checkpoints/3D_unet_g3_h3_crop_'+str(chosen_crop_types_list[0])+'_'+str(chosen_crop_types_list[1])+'_'+str(chosen_crop_types_list[2])+'_epoch_'+str(num_epochs)+'.h5'

        #else:
        #my_model_1 = load_model('./storage/saved_model/3D_unet_g_'+str(g)+'_h_'+str(h)+'_labels_'+vista_crop_dict[chosen_crop_types_list[0]]+'_'+vista_crop_dict[chosen_crop_types_list[1]]+'_'+vista_crop_dict[chosen_crop_types_list[2]]+'_epoch_'+str(num_epochs)+'.h5', compile=False)

        test_pred1 = my_model_1.predict(test_img_input1)
        #print("test_pred1.shape", test_pred1.shape)
        del my_model_1
        gc.collect()
        test_prediction1 = np.argmax(test_pred1, axis=4)[0,:,:,:]

        #print("test_prediction1.shape", test_prediction1.shape)
        #plt.imshow(test_prediction1[0,:,:])
        #plt.savefig("/data1/chethan/stelar_3dunet/vista_patch_exp0/ensamble_results/general_tests/pre"+str(ll)+".png")
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
    all_test_mode.append(test_mode)

    ground_truth_flattened = ground_truth_flattened.astype(np.uint8)
    all_ground_truth_flattened.append(ground_truth_flattened)

    test_mode_agg = np.stack(all_test_mode)
    all_ground_truth_flattened_agg = np.stack(all_ground_truth_flattened)

    #print("test_mode_agg.shape", test_mode_agg.shape)
    #print("all_ground_truth_flattened_agg.shape", all_ground_truth_flattened_agg.shape)

    tifffile.imwrite(prefix+'/vista_patch_exp0/saved_unaggeregated_outputs_f1/test_mode_agg_'+season+'_.tif', test_mode_agg)
    tifffile.imwrite(prefix+'/vista_patch_exp0/saved_unaggeregated_outputs_f1/all_ground_truth_flattened_agg'+season+'_.tif', all_ground_truth_flattened_agg)





aggregated_predicted = np.zeros((10048, 10048))
aggregated_ground_truth = np.zeros((10048, 10048))



#predicted = io.imread('/data1/chethan/stelar_3dunet/vista_patch_exp0/saved_unaggeregated_outputs/test_mode_agg_'+season+'_.tif')
#ground_truth = io.imread('/data1/chethan/stelar_3dunet/vista_patch_exp0/saved_unaggeregated_outputs/all_ground_truth_flattened_agg'+season+'_.tif')




all_processed_LAI_stacked = []
all_labels = []
k = 0
for i in range(aggregated_predicted.shape[-1]//64):
    for j in range(aggregated_predicted.shape[-2]//64):
        #print("i", i)
        #print("j", j)
        #print("all_processed_LAI[:, 0:64, 0:64].shape", aggregated_predicted[i*64:(i*64)+64, j*64:(j*64)+64].shape)

        aggregated_predicted[i*64:(i*64)+64, j*64:(j*64)+64] = test_mode_agg[k]
        aggregated_ground_truth[i*64:(i*64)+64, j*64:(j*64)+64] = all_ground_truth_flattened_agg[k]
        k+=1


print("aggregated_predicted.shape", aggregated_predicted.shape)
print("aggregated_ground_truth.shape", aggregated_ground_truth.shape)

aggregated_predicted = aggregated_predicted[:10002, :10002]
aggregated_ground_truth = aggregated_ground_truth[:10002, :10002]


test_mode_c = get_labels_in_color(aggregated_predicted)
ground_truth_c = get_labels_in_color(aggregated_ground_truth)


plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
plt.imshow(test_mode_c)
plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1/'+season+'predicted_.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
plt.imshow(ground_truth_c)
plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1/'+season+'ground_truth_.png', bbox_inches='tight')
plt.close()





# Parameters from RHD header
ulx = 704855.0        # upper-left X
uly = 4995145.0       # upper-left Y
pixel_size = 10.0     # resolution
crs_code = 'EPSG:32630'

# Calculate transform
transform = from_origin(ulx, uly, pixel_size, pixel_size)

# Save as GeoTIFF


if season=="Feb_Aug":

    plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
    plt.imshow(test_mode_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/Feb_Aug_predicted_.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
    plt.imshow(ground_truth_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/Feb_Aug_ground_truth_.png', bbox_inches='tight')
    plt.close()

    with rasterio.open(
        'aggregated_predicted_Feb_Aug.tif',
        'w',
        driver='GTiff',
        height=aggregated_predicted.shape[0],
        width=aggregated_predicted.shape[1],
        count=1,
        dtype=aggregated_predicted.dtype,
        crs=crs_code,
        transform=transform
    ) as dst:
        dst.write(aggregated_predicted, 1)

    # Path to the saved file
    filename = 'aggregated_predicted_Feb_Aug.tif'

    # Open the file in read mode
    with rasterio.open(filename, 'r') as src:
        data = src.read(1)  # Read the first (and only) band
        print("Shape:", data.shape)
        print("Data type:", data.dtype)



if season=="May_Aug":


    plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
    plt.imshow(test_mode_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/May_Aug_predicted_.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
    plt.imshow(ground_truth_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/May_Aug_ground_truth_.png', bbox_inches='tight')
    plt.close()

    with rasterio.open(
        'aggregated_predicted_May_Aug.tif',
        'w',
        driver='GTiff',
        height=aggregated_predicted.shape[0],
        width=aggregated_predicted.shape[1],
        count=1,
        dtype=aggregated_predicted.dtype,
        crs=crs_code,
        transform=transform
    ) as dst:
        dst.write(aggregated_predicted, 1)

    # Path to the saved file
    filename = 'aggregated_predicted_May_Aug.tif'

    # Open the file in read mode
    with rasterio.open(filename, 'r') as src:
        data = src.read(1)  # Read the first (and only) band
        print("Shape:", data.shape)
        print("Data type:", data.dtype)


if season=="Jun_Oct":

    plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
    plt.imshow(test_mode_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/Jun_Oct_predicted_.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
    plt.imshow(ground_truth_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/Jun_Oct_ground_truth_.png', bbox_inches='tight')
    plt.close()


    with rasterio.open(
        'aggregated_predicted_Jun_Oct.tif',
        'w',
        driver='GTiff',
        height=aggregated_predicted.shape[0],
        width=aggregated_predicted.shape[1],
        count=1,
        dtype=aggregated_predicted.dtype,
        crs=crs_code,
        transform=transform
    ) as dst:
        dst.write(aggregated_predicted, 1)

    # Path to the saved file
    filename = 'aggregated_predicted_Jun_Oct.tif'

    # Open the file in read mode
    with rasterio.open(filename, 'r') as src:
        data = src.read(1)  # Read the first (and only) band
        print("Shape:", data.shape)
        print("Data type:", data.dtype)



if season=="Jan_Aug":

    plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
    plt.imshow(test_mode_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/Jan_Aug_predicted_.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
    plt.imshow(ground_truth_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/Jan_Aug_ground_truth_.png', bbox_inches='tight')
    plt.close()


    with rasterio.open(
        'aggregated_predicted_Jan_Aug.tif',
        'w',
        driver='GTiff',
        height=aggregated_predicted.shape[0],
        width=aggregated_predicted.shape[1],
        count=1,
        dtype=aggregated_predicted.dtype,
        crs=crs_code,
        transform=transform
    ) as dst:
        dst.write(aggregated_predicted, 1)

    # Path to the saved file
    filename = 'aggregated_predicted_Jan_Aug.tif'

    # Open the file in read mode
    with rasterio.open(filename, 'r') as src:
        data = src.read(1)  # Read the first (and only) band
        print("Shape:", data.shape)
        print("Data type:", data.dtype)
