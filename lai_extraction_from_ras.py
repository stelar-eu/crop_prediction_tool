

from functions import extract_LAI_from_RAS_file, explore_image, extract_all_LAI_from_RAS_file, extract_spec_LAI_from_RAS_file, get_cluster_length
import matplotlib.pyplot as plt
import numpy as np
datapath = './dataset/france2/lai_ras/'



import argparse

parser = argparse.ArgumentParser(description='Enter crop type numbers in order')

parser.add_argument('--image_length', type=float, default=10002, help='input season')
parser.add_argument('--image_width', type=float, default=10002, help='input season')



args = parser.parse_args()
image_length = args.image_length
image_width = args.image_width



import glob
filepaths = glob.glob('./dataset/france2/lai_ras/*.RAS')

filepaths.sort()

for datapath_filename in filepaths:
    print("datapath_filename", datapath_filename)
    cluster_len = get_cluster_length(datapath_filename, image_length, image_width)
    for cluster_ind in range(cluster_len):
        test = extract_spec_LAI_from_RAS_file(datapath_filename, cluster_ind, image_length, image_width)
        test[test<0] = 0  
        print("datapath_filename[-14:-4]+'_measure_'+str(i) : ", datapath_filename[-14:-4]+'_measure_'+str(cluster_ind).zfill(2))
        np.save('./dataset/france2/processed_lai_npy1/'+datapath_filename[-14:-4]+'_measure_'+str(cluster_ind).zfill(2)+'.npy', test)

