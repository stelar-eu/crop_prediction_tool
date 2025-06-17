# Imports

import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
from keras.models import load_model
from scipy.stats import mode
import tensorflow as tf
import segmentation_models_3D as sm
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import glob
import gc
import rasterio
from rasterio.transform import from_origin
import argparse
import gc
import tifffile

parser = argparse.ArgumentParser(description='Enter crop type numbers in order')

parser.add_argument('--season', type=str, default=0, help='input season')

args = parser.parse_args()
season = args.season

device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))

prefix = '/app/'
# Parameters from RHD header
ulx = 704855.0        # upper-left X
uly = 4995145.0       # upper-left Y
pixel_size = 10.0     # resolution
crs_code = 'EPSG:32630'

# address of time series LAI
filepaths = glob.glob(prefix+'/dataset/france2/processed_lai_npy/*.npy')
filepaths.sort()

# Functions

def slice_and_stack_cubes(data, cube_size=64):
    z_dim, x_dim, y_dim = data.shape
    cubes = []
    for x in range(0, x_dim, cube_size):
        for y in range(0, y_dim, cube_size):
            if x + cube_size <= x_dim and y + cube_size <= y_dim:
                cube = data[:, x:x + cube_size, y:y + cube_size]
                cubes.append(cube)
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
    for x in range(0, x_dim, slice_size):
        for y in range(0, y_dim, slice_size):
            if x + slice_size <= x_dim and y + slice_size <= y_dim:
                slice = labels[x:x + slice_size, y:y + slice_size]
                slices.append(slice)
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



BACKBONE = 'vgg16'  
color_map = {10: [0, 0, 0], 0: [0, 0, 0], 11: [0, 255, 0], 12: [0, 0, 255], 13: [255, 255, 0], 14: [255, 165, 0], 15: [255, 0, 255], 16: [0, 255, 255], 17: [128, 0, 128], 18: [128, 128, 0], 19: [0, 128, 0], 20: [128, 0, 0], 21: [0, 0, 128], 22: [128, 128, 128], 23: [0, 128, 128], 24: [255, 0, 0], 25: [255, 255, 255], 26: [192, 192, 192], 27: [139, 0, 0], 28: [0, 100, 0], 29: [0, 0, 139], 30: [255, 215, 0], 31: [255, 140, 0], 32: [139, 0, 139], 33: [0, 206, 209], 34: [75, 0, 130], 35: [85, 107, 47], 36: [34, 139, 34], 37: [165, 42, 42], 38: [70, 130, 180], 39: [169, 169, 169], 40: [32, 178, 170], 41: [47, 79, 79], 42: [245, 245, 245], 43: [105, 105, 105], 44: [205, 92, 92], 45: [50, 205, 50], 46: [65, 105, 225], 47: [255, 223, 0], 48: [255, 99, 71], 49: [186, 85, 211], 50: [0, 191, 255], 51: [192, 192, 192]}
crop_types_all_list = [ 10, 11,  12,  13,  14,  15,  17,  18,  19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 37, 38, 40, 42, 43, 44, 45, 46, 47, 50, 51]
vista_crop_dict = {0:'NA', 10:'NA' , 11: 'ALFALFA', 12: 'BEET', 13: 'CLOVER', 14: 'FLAX', 15: 'FLOWERING_LEGUMES', 16: 'FLOWERS', 17: 'FOREST', 18: 'GRAIN_MAIZE', 19: 'GRASSLAND', 20: 'HOPS', 21: 'LEGUMES', 22: 'VISTA_NA', 23: 'PERMANENT_PLANTATIONS', 24: 'PLASTIC', 25: 'POTATO', 26: 'PUMPKIN', 27: 'RICE', 28: 'SILAGE_MAIZE', 29: 'SOY', 30: 'SPRING_BARLEY', 31: 'SPRING_OAT', 32: 'SPRING_OTHER_CEREALS', 33: 'SPRING_RAPESEED', 34: 'SPRING_RYE', 35: 'SPRING_SORGHUM', 36: 'SPRING_SPELT', 37: 'SPRING_TRITICALE', 38: 'SPRING_WHEAT', 39: 'SUGARBEET', 40: 'SUNFLOWER', 41: 'SWEET_POTATOES', 42: 'TEMPORARY_GRASSLAND', 43: 'WINTER_BARLEY', 44: 'WINTER_OAT', 45: 'WINTER_OTHER_CEREALS', 46: 'WINTER_RAPESEED', 47: 'WINTER_RYE', 48: 'WINTER_SORGHUM', 49: 'WINTER_SPELT', 50: 'WINTER_TRITICALE', 51: 'WINTER_WHEAT'}
labels = np.load(prefix+'/storage/full_mast/vista_labes_aligned.npy').astype(np.uint8)


if(season=="Feb_Aug"):
    chosen_crop_types_list_list_models = [[43, 46, 51], [44, 47, 50]]
    time_strip_ind = (len(filepaths) // 12) * 1
    chosen_season_crops = [10, 43, 44, 46, 47, 50, 51]

if(season=="May_Aug"):
    chosen_crop_types_list_list_models = [[12, 25, 30], [31, 33, 38]]
    time_strip_ind = (len(filepaths) // 12) * 3
    chosen_season_crops = [10, 12, 25, 30, 31, 33, 38]

if(season=="Jun_Oct"):
    chosen_crop_types_list_list_models = [[18, 19, 40], [17, 28, 29]]
    time_strip_ind = (len(filepaths) // 12) * 4
    chosen_season_crops = [10, 18, 19, 40, 17, 28, 29]

if(season=="Jan_Aug"):
    chosen_crop_types_list_list_models = [[14, 17, 19]]
    time_strip_ind = (len(filepaths) // 12) * 0
    chosen_season_crops = [10, 14, 17, 19]


# Time series creation 

considered_filepaths = filepaths[time_strip_ind:time_strip_ind+64]   

numpy_array_all = []
pad_height = 46
pad_width = 46
k = 0
for filepath in considered_filepaths:
    numpy_array = np.load(filepath).astype(np.float32)
    tf_tensor = tf.convert_to_tensor(numpy_array)  # defaults to float32 already
    tf_tensor_padded = np.pad(tf_tensor, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    tf_tensor_padded = tf.identity(tf_tensor_padded)  # placeholder; TF automatically uses GPU if available
    k+=1
    numpy_array_all.append(tf_tensor_padded)

all_processed_LAI = np.stack(numpy_array_all, axis=0)
del numpy_array_all
gc.collect()
labels = tf.convert_to_tensor(labels)  # defaults to float32 already
labels = np.pad(labels, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
labels = tf.identity(labels)  # placeholder; TF automatically uses GPU if available
all_processed_LAI = tf.convert_to_tensor(all_processed_LAI, dtype=tf.float32)

print("all_processed_LAI.device", all_processed_LAI.device)

# Slicing the spatio-temporal test data

all_processed_LAI_stacked = []
all_labels = []
for i in range(all_processed_LAI.shape[-1]//64):
    for j in range(all_processed_LAI.shape[-2]//64):
        all_processed_LAI_stacked.append(all_processed_LAI[:, i*64:(i*64)+64, j*64:(j*64)+64])
        all_labels.append(labels[i*64:(i*64)+64, j*64:(j*64)+64])
del all_processed_LAI
gc.collect()


sliced_cubes = np.stack(all_processed_LAI_stacked, axis=0)
sliced_cubes = tf.convert_to_tensor(sliced_cubes, dtype=tf.float32)

sliced_labels = np.stack(all_labels, axis=0)
sliced_labels = tf.convert_to_tensor(sliced_labels, dtype=tf.float32)

del all_labels
del labels
gc.collect()


X_test = np.stack((sliced_cubes,)*3, axis=-1)
sliced_labels = np.repeat(sliced_labels[:, np.newaxis, :, :], repeats=64, axis=1)
y_test_1 = sliced_labels

del sliced_cubes
del sliced_labels
gc.collect()



# Inference

check = 0
all_test_mode = []
all_ground_truth_flattened = []
for test_img_number in range(len(X_test)):

    chosen_subset_for_test_set = 0
    stacked_test_preds = []
    ground_truth_1 = y_test_1[test_img_number-1] + 10

    for k in crop_types_all_list:
        if not(k in chosen_season_crops):
            ground_truth_1[ground_truth_1==k]=0

    ensambled_ground_truth = np.zeros((64, 64))
    ensambled_result_image = np.zeros((64, 64))
    ensambled_result_image_gen = np.zeros((64, 64, 64))
    test_img = X_test[test_img_number-1]
    test_img_input1=np.expand_dims(test_img, 0)
    
    ground_truth_flattened = ground_truth_1[0,:,:]

    ll = 0 
    for chosen_crop_types_list in chosen_crop_types_list_list_models:
        gc.collect()
        my_model_1 = sm.Unet(BACKBONE, input_shape=(64, 64, 64, 3), classes=4, encoder_weights=None, activation='softmax')
        my_model_1.load_weights(prefix+'/checkpoints_f1/3D_unet_g4_h4_crop_'+str(chosen_crop_types_list[0]-10)+'_'+str(chosen_crop_types_list[1]-10)+'_'+str(chosen_crop_types_list[2]-10)+'_epoch_f.h5', by_name=True, skip_mismatch=True)
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
    all_test_mode.append(test_mode)

    ground_truth_flattened = ground_truth_flattened.astype(np.uint8)
    all_ground_truth_flattened.append(ground_truth_flattened)

    test_mode_agg = np.stack(all_test_mode)
    all_ground_truth_flattened_agg = np.stack(all_ground_truth_flattened)


# Rebuilding to original size
aggregated_predicted = np.zeros((10048, 10048))
aggregated_ground_truth = np.zeros((10048, 10048))
all_processed_LAI_stacked = []
all_labels = []
k = 0
for i in range(aggregated_predicted.shape[-1]//64):
    for j in range(aggregated_predicted.shape[-2]//64):
        aggregated_predicted[i*64:(i*64)+64, j*64:(j*64)+64] = test_mode_agg[k]
        aggregated_ground_truth[i*64:(i*64)+64, j*64:(j*64)+64] = all_ground_truth_flattened_agg[k]
        k+=1

print("aggregated_predicted.shape", aggregated_predicted.shape)
print("aggregated_ground_truth.shape", aggregated_ground_truth.shape)

aggregated_predicted = aggregated_predicted[:10002, :10002]
aggregated_ground_truth = aggregated_ground_truth[:10002, :10002]


tifffile.imwrite(prefix+'/vista_patch_exp0/predicted_outputs/predicted_'+season+'_.tif', aggregated_predicted)
tifffile.imwrite(prefix+'/vista_patch_exp0/predicted_outputs/ground_truth_'+season+'_.tif', aggregated_ground_truth)


test_mode_c = get_labels_in_color(aggregated_predicted)
ground_truth_c = get_labels_in_color(aggregated_ground_truth)

transform = from_origin(ulx, uly, pixel_size, pixel_size)

#Save as GeoTIFF
if season=="Feb_Aug":

    plt.figure(figsize=(50, 50))  
    plt.imshow(test_mode_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/Feb_Aug_predicted_.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(50, 50)) 
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

    filename = 'aggregated_predicted_Feb_Aug.tif'

    with rasterio.open(filename, 'r') as src:
        data = src.read(1) 
        print("Shape:", data.shape)
        print("Data type:", data.dtype)



if season=="May_Aug":


    plt.figure(figsize=(50, 50))  
    plt.imshow(test_mode_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/May_Aug_predicted_.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(50, 50))  
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

    filename = 'aggregated_predicted_May_Aug.tif'

    with rasterio.open(filename, 'r') as src:
        data = src.read(1)  
        print("Shape:", data.shape)
        print("Data type:", data.dtype)


if season=="Jun_Oct":

    plt.figure(figsize=(50, 50))  
    plt.imshow(test_mode_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/Jun_Oct_predicted_.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(50, 50))  
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

    filename = 'aggregated_predicted_Jun_Oct.tif'

    with rasterio.open(filename, 'r') as src:
        data = src.read(1) 
        print("Shape:", data.shape)
        print("Data type:", data.dtype)



if season=="Jan_Aug":

    plt.figure(figsize=(50, 50))  
    plt.imshow(test_mode_c)
    plt.savefig(prefix+'/vista_patch_exp0/aggregated_plots_f1_gt/Jan_Aug_predicted_.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(50, 50))  
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

    filename = 'aggregated_predicted_Jan_Aug.tif'

    with rasterio.open(filename, 'r') as src:
        data = src.read(1)  
        print("Shape:", data.shape)
        print("Data type:", data.dtype)
