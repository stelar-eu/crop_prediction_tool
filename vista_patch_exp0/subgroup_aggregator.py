
import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
import tifffile


#Make sure the GPU is available. 
import tensorflow as tf

device_name = tf.test.gpu_device_name()
'''if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')'''
print('Found GPU at: {}'.format(device_name))




'''





Feb_Aug:

cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 34 --crop_2 37 --crop_3 40
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 33 --crop_2 36 --crop_3 41


May_Aug:

cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 2 --crop_2 15 --crop_3 20
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 21 --crop_2 23 --crop_3 28


Jun_Oct crops

cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 8 --crop_2 9 --crop_3 30
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 7 --crop_2 18 --crop_3 19


Jan_Aug 

cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 4 --crop_2 7 --crop_3 9


'''



import segmentation_models_3D as sm


from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



'''physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)'''



def subgroup_aggregator(cr_1, cr_2, cr_3):

  vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}

  '''import argparse

  parser = argparse.ArgumentParser(description='Enter crop type numbers in order')
  parser.add_argument('--crop_1', type=int, default=1, help='Select crop type')
  parser.add_argument('--crop_2', type=int, default=2, help='Select crop type')
  parser.add_argument('--crop_3', type=int, default=3, help='Select crop type')

  args = parser.parse_args()

  cr_1 = args.crop_1
  cr_2 = args.crop_2
  cr_3 = args.crop_3'''

  '''cr_1 = 4
  cr_2 = 7
  cr_3 = 9'''


  chosen_crop_types_list = [cr_1, cr_2, cr_3]
  crop_types_all_list = [ 1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 27, 28, 30, 32, 33, 34, 35, 36, 37, 40, 41]
  sampling_group_fractions = [1.0, 1.0, 1.0]


  all_input_img_f = []
  all_input_mask_f = []
  all_input_coords_f = []

  counted = 0
  for crop_no in chosen_crop_types_list:
      chosen_crop_type = vista_crop_dict[crop_no]

      input_img_f = io.imread('./storage/per_crop_data_labels_test/'+vista_crop_dict[crop_no]+'/train'+vista_crop_dict[crop_no]+'n.tif')#[:9000]
      input_mask_f = io.imread('./storage/per_crop_data_labels_test/'+vista_crop_dict[crop_no]+'/lab'+vista_crop_dict[crop_no]+'n.tif').astype(np.uint8)#[:9000]

      bis = int(len(input_img_f)*sampling_group_fractions[counted]) - 2

      input_img_f = input_img_f[:bis]
      input_mask_f = input_mask_f[:bis]


      all_input_img_f.append(input_img_f)
      all_input_mask_f.append(input_mask_f)

      counted+=1    
  all_input_img_f = np.concatenate((all_input_img_f), axis=0)
  all_input_mask_f = np.concatenate((all_input_mask_f), axis=0)



  input_img = all_input_img_f
  input_mask = all_input_mask_f

  del all_input_img_f
  del all_input_mask_f
  del input_img_f
  del input_mask_f

  input_mask = np.repeat(input_mask[:, np.newaxis, :, :], repeats=64, axis=1)

  lai_uniques = 0
  n_classes=4
  train_img = np.stack((input_img,)*3, axis=-1)
  X_train, X_test, y_train, y_test_all = train_test_split(train_img, input_mask, test_size = 0.9, random_state = 0)


  del X_train
  del y_train

  X_test = X_test[:1000]
  y_test_all = y_test_all[:1000]

  print("X_test.shape", X_test.shape)
  print("y_test_all.shape", y_test_all.shape)

  tifffile.imsave('./storage/test_sets_of_subsets_all/X_test'+'_contains'+vista_crop_dict[cr_1]+'_'+vista_crop_dict[cr_2]+'_'+vista_crop_dict[cr_3]+'_.tif', X_test)
  tifffile.imsave('./storage/test_sets_of_subsets_all/y_test_all'+'_contains'+vista_crop_dict[cr_1]+'_'+vista_crop_dict[cr_2]+'_'+vista_crop_dict[cr_3]+'_.tif', y_test_all)
