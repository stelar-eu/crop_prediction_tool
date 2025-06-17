


'''

export CUDA_VISIBLE_DEVICES=0
conda deactivate
conda deactivate
conda activate /home/luser/miniforge3/envs/stcon4
cd stelar_3dunet/
/home/luser/miniforge3/envs/stcon4/bin/python label_extraction.py 

'''


'''
new packages to be added 

conda install conda-forge::geopandas

conda install conda-forge::opencv



'''


#use the environment newgpuenv

# Do all the required imports

#import pandas as pd
import numpy as np
import geopandas
import matplotlib.pyplot as plt
#import shapefile as shp
import os

#from pystac_client import Client
#from shapely.geometry import Point
#import shapely.wkt
#import shapely.ops
#import rioxarray
import rasterio
import numpy as np
from numpy import inf
from PIL import Image
import math
#import functions
import pyproj
from shapely.ops import transform
from pyproj import transform as transform2
from pyproj import Proj
from matplotlib import pyplot

import matplotlib.path as mpl_path
import cv2



os.environ['USE_PYGEOS'] = '0'


prefix = '/app/'

#labels = np.load(prefix+'/storage/full_mast/vista_labes_aligned.npy').astype(np.uint8)


# Load the shape files

shpPath = prefix+'/dataset/shape_files_from_Eurocrops/dataset5_france/FR_2018/'
shpName = 'FR_2018_EC21'
csvPath = prefix+'/dataset/shape_files_from_Eurocrops/dataset5_france/'
csvName = 'fr_2018.csv'


ALFALFA = [3301090301]
BEET=[3301050000, 3301290200, 3301290400]
CLOVER=[3301090303]
FLAX=[3301060701, 3301060702]
FLOWERING_LEGUMES=[3301020700]
FLOWERS=[3301080000]
FOREST=[3306000000, 3306010000, 3306020000, 3306030000, 3306040000, 3306050000, 3306060000, 3306070000, 3306080000, 3306980000, 3306990000]
GRAIN_MAIZE=[3301010600, 3301010699]
GRASSLAND=[3302000000]
HOPS=[3301060200]
LEGUMES=[3301020100, 3301020500, 3301020600, 3301029900, 3301090300, 3301090302, 3301090304, 3301090305, 3301090398]
NA=[3000000000, 3300000000, 3301000000, 3301010000, 3301010100, 3301010200, 3301010300, 3301010400, 3301010500, 3301010800, 3301010900, 3301011000, 3301020000, 3301060400, 3301060700, 3301090000]
PERMANENT_PLANTATIONS=[3303010000, 3303060000]
PLASTIC=[3305000000]
POTATO=[3301030000]
PUMPKIN=[3301140400]
RICE=[3301010700, 3301010799]
SILAGE_MAIZE=[3301090400]
SOY=[3301160000]
SPRING_BARLEY=[3301010402]
SPRING_OAT=[3301010502]
SPRING_OTHER_CEREALS=[3301011102, 3301011202, 3301011302, 3301011502, 3301011503]
SPRING_RAPESEED=[3301060402, 3301060403]
SPRING_RYE=[3301010302]
SPRING_SORGHUM=[3301010902]
SPRING_SPELT=[3301011002]
SPRING_TRITICALE=[3301010802]
SPRING_WHEAT=[3301010102, 3301010202]
SUGARBEET=[3301290700]
SUNFLOWER=[3301060500]
SWEET_POTATOES=[3301040000]
TEMPORARY_GRASSLAND=[3301090100, 3301090200, 3301090201, 3301090202, 3301090203, 3301090204, 3301090205, 3301090206, 3301090207, 3301090208, 3301090209]
WINTER_BARLEY=[3301010401]
WINTER_OAT=[3301010501]
WINTER_OTHER_CEREALS=[3301011101, 3301011201, 3301011301, 3301011501]
WINTER_RAPESEED=[3301060401]
WINTER_RYE=[3301010301]
WINTER_SORGHUM=[3301010901]
WINTER_SPELT=[3301011001]
WINTER_TRITICALE=[3301010801]
WINTER_WHEAT=[3301010101, 3301010201]



all_crops = [ALFALFA, BEET, CLOVER, FLAX, FLOWERING_LEGUMES, FLOWERS, FOREST, GRAIN_MAIZE, GRASSLAND, HOPS, LEGUMES, NA, PERMANENT_PLANTATIONS, PLASTIC, POTATO, PUMPKIN, RICE, SILAGE_MAIZE, SOY, SPRING_BARLEY, SPRING_OAT, SPRING_OTHER_CEREALS, SPRING_RAPESEED, SPRING_RYE, SPRING_SORGHUM, SPRING_SPELT, SPRING_TRITICALE, SPRING_WHEAT, SUGARBEET, SUNFLOWER, SWEET_POTATOES, TEMPORARY_GRASSLAND, WINTER_BARLEY, WINTER_OAT, WINTER_OTHER_CEREALS, WINTER_RAPESEED, WINTER_RYE, WINTER_SORGHUM, WINTER_SPELT, WINTER_TRITICALE, WINTER_WHEAT]

all_crops_flat = [item for sublist in all_crops for item in sublist]

# this is from VISTA readme files of LAI
Lai_eastings = np.array([704855.0000, 804875.0000])
Lai_northings = np.array([4895125.0000, 4995145.0000])


shpfile = geopandas.read_file(shpPath+shpName+'.shp')

print("shpfile.crs", shpfile.crs)


#drop NaNs from geometry EC_hcat_c
shpfile = shpfile.dropna(subset=['EC_hcat_c'])
#convert the EC_hcat_c column to int
shpfile['EC_hcat_c'] = shpfile['EC_hcat_c'].astype(int)


#b_minx, b_miny, b_maxx, b_maxy = 418571.25455360627, 4574362.735985591, 1462315.824437159, 5721834.6795369005
project = pyproj.Transformer.from_proj(
    pyproj.Proj(init='EPSG:2154'), # source coordinate system
    pyproj.Proj(init='EPSG:32630')) # destination coordinate system


#label_space = np.zeros((eurocrops_x_pixel_size_p, eurocrops_y_pixel_size_p))


mask = np.zeros((10002, 10002), dtype=np.uint8)


print("started")

for i in range(len(shpfile)):
    polygon = shpfile.iloc[i]['geometry']
    hcat = shpfile.iloc[i]['EC_hcat_c']
    #print("hcat ", hcat)
    if(hcat in all_crops_flat):
        polygon = transform(project.transform, polygon)
        #print("polygon", polygon)
        #polygon = shapely.wkt.loads(polygon)
        if polygon.geom_type == 'Polygon':
            coordinates = list(polygon.exterior.coords)
            #print("coordinates", coordinates)
            np_coordinates = np.array(coordinates)
            bring = np.array([Lai_eastings[0], Lai_northings[0]])
            reduced_coords = (coordinates - bring) 
            #print("reduced_coords", reduced_coords)
            check = ((  reduced_coords / np.array([Lai_eastings[1]-Lai_eastings[0], Lai_northings[1]-Lai_northings[0]])   )*10002).astype(int)
            if(not  (  (False in (check > 0 )) or (False in (check < 10002))  )  ):
                for i,crop_row in enumerate(all_crops):
                    if(hcat in crop_row):
                        #print("check", check)
                        #print("False in check")
                        cv2.fillPoly(mask, [check], i)      
                        break      


mask1 = np.flip(mask, 0 )


np.save(prefix+'/dataset/euro_mask/vista_labes_image.npy', mask1)


def get_labels_in_color(groud_truth_image):
    color_map = {10: [0, 0, 0], 0: [0, 0, 0], 11: [0, 255, 0], 12: [0, 0, 255], 13: [255, 255, 0], 14: [255, 165, 0], 15: [255, 0, 255], 16: [0, 255, 255], 17: [128, 0, 128], 18: [128, 128, 0], 19: [0, 128, 0], 20: [128, 0, 0], 21: [0, 0, 128], 22: [128, 128, 128], 23: [0, 128, 128], 24: [255, 0, 0], 25: [255, 255, 255], 26: [192, 192, 192], 27: [139, 0, 0], 28: [0, 100, 0], 29: [0, 0, 139], 30: [255, 215, 0], 31: [255, 140, 0], 32: [139, 0, 139], 33: [0, 206, 209], 34: [75, 0, 130], 35: [85, 107, 47], 36: [34, 139, 34], 37: [165, 42, 42], 38: [70, 130, 180], 39: [169, 169, 169], 40: [32, 178, 170], 41: [47, 79, 79], 42: [245, 245, 245], 43: [105, 105, 105], 44: [205, 92, 92], 45: [50, 205, 50], 46: [65, 105, 225], 47: [255, 223, 0], 48: [255, 99, 71], 49: [186, 85, 211], 50: [0, 191, 255], 51: [192, 192, 192]}

    groud_truth_color_image = np.zeros(groud_truth_image.shape + (3,), dtype=np.uint8)
    for i in range(groud_truth_image.shape[0]):
        for j in range(groud_truth_image.shape[1]):
            segment_id_gt = groud_truth_image[i, j]
            groud_truth_color_image[i, j] = color_map[segment_id_gt]
    return groud_truth_color_image



mask_color = get_labels_in_color(mask1)

plt.figure(figsize=(50, 50))  # Increase size: width=10 inches, height=10 inches
plt.imshow(mask_color)
plt.savefig(prefix+'/dataset/euro_mask/vista_labes_image.png', bbox_inches='tight')
plt.close()