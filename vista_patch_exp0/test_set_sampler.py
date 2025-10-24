#from functions import extract_LAI_from_RAS_file, explore_image, extract_all_LAI_from_RAS_file

'''import torch
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
temporal_batches = torch.tensor([]).to(device)'''



import matplotlib.pyplot as plt
#import torch
import numpy as np
#datapath = './dataset/france/lai_ras/'
import random
import glob
import tifffile
import logging



# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


filepaths = glob.glob('./dataset/france2/processed_lai_npy/*.npy')
filepaths.sort()
#print("time paths", filepaths)
# put a universal seed for all random initialiazitations


'''



Winter crops (Feb_Aug)
	•	WINTER_BARLEY (33)
	•	WINTER_OAT (34)
	•	WINTER_OTHER_CEREALS (35)
	•	WINTER_RAPESEED (36)
	•	WINTER_RYE (37)
	•	WINTER_TRITICALE (40)
	•	WINTER_WHEAT (41)
    chosen_crop_types_list_list = [[33, 36, 41], [34, 37, 40]]

    in the landscape (g = 0, h = 0) there is no WINTER_OTHER_CEREALS (35)

        chosen_crop_types_list_list = [[33, 36, 41], [34, 37, 40]]


cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 33 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 36 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 41 --season Feb_Aug


python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 34 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 37 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 40 --season Feb_Aug





Spring crops (May_Aug)
•	BEET(2)
•	POTATO(15)
•	SPRING_BARLEY(20)    
•	SPRING_OAT(21)
•	SPRING_RAPESEED(23)
•	SPRING_WHEAT(28)


cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 2 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 15 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 20 --season May_Aug

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 21 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 23 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 28 --season May_Aug




summer_autumn crops (Jun_Oct)

	• GRAIN_MAIZE(8)
    • SUNFLOWER(30)
    • GRASSLAND(9)

    • SILAGE_MAIZE(18)
    • SOY(19)
    • FOREST(7)

cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 8 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 30 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 9 --season Jun_Oct

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 18 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 19 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 7 --season Jun_Oct


    



winter_spring_summer (Jan_Aug)

FLAX((4)
FOREST(7)
GRASSLAND(9)


cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 4 --season Jan_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 7 --season Jan_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 9 --season Jan_Aug




'''

vista_crop_dict = { 0:'NA' , 1: 'ALFALFA', 2: 'BEET', 3: 'CLOVER', 4: 'FLAX', 5: 'FLOWERING_LEGUMES', 6: 'FLOWERS', 7: 'FOREST', 8: 'GRAIN_MAIZE', 9: 'GRASSLAND', 10: 'HOPS', 11: 'LEGUMES', 12: 'VISTA_NA', 13: 'PERMANENT_PLANTATIONS', 14: 'PLASTIC', 15: 'POTATO', 16: 'PUMPKIN', 17: 'RICE', 18: 'SILAGE_MAIZE', 19: 'SOY', 20: 'SPRING_BARLEY', 21: 'SPRING_OAT', 22: 'SPRING_OTHER_CEREALS', 23: 'SPRING_RAPESEED', 24: 'SPRING_RYE', 25: 'SPRING_SORGHUM', 26: 'SPRING_SPELT', 27: 'SPRING_TRITICALE', 28: 'SPRING_WHEAT', 29: 'SUGARBEET', 30: 'SUNFLOWER', 31: 'SWEET_POTATOES', 32: 'TEMPORARY_GRASSLAND', 33: 'WINTER_BARLEY', 34: 'WINTER_OAT', 35: 'WINTER_OTHER_CEREALS', 36: 'WINTER_RAPESEED', 37: 'WINTER_RYE', 38: 'WINTER_SORGHUM', 39: 'WINTER_SPELT', 40: 'WINTER_TRITICALE', 41: 'WINTER_WHEAT'}



import argparse

parser = argparse.ArgumentParser(description='Crop speicific LAI and labels sampling')
parser.add_argument('--chosen_crop_types', type=int, default=3, help='Select crop type')
parser.add_argument('--season', type=str, default=0, help='input season')


args = parser.parse_args()


chosen_crop_types = args.chosen_crop_types
season = args.season


if season=="Feb_Aug":
    time_strip_inds = [(len(filepaths) // 12) * 1, ((len(filepaths) // 12) * 1) + 10 ]  # for winter 
if season=="May_Aug":
    time_strip_inds = [ (len(filepaths) // 12) * 3, ( (len(filepaths) // 12) * 3)+5 ] # for spring
if season=="Jun_Oct":
    time_strip_inds = [(len(filepaths) // 12) * 4, ((len(filepaths) // 12) * 4)-5] # for summer_autumn
if season=="Jan_Aug":
    time_strip_inds = [(len(filepaths) // 12) * 0, ((len(filepaths) // 12) * 0) + 1] # for summer_autumn


logging.info(f"Selected crop type: {vista_crop_dict[chosen_crop_types]}")


labels = np.load('./storage/full_mast/vista_labes_aligned.npy').astype(np.uint8)


if(chosen_crop_types==23):
    intra_loop_length = 667//6
else:
    intra_loop_length = 500//6

all_lai_list = [] # new
all_labels_list = [] # new

for time_strip_ind in time_strip_inds:

    temporal_batches = np.array([]) # new 
    temporal_batches_list = []
    spatial_label_batches = np.array([]) # mew
    spatial_label_batches_list = []

    print("len(filepaths)", len(filepaths))
    print("time_strip_ind", time_strip_ind)

    considered_filepaths = filepaths[time_strip_ind:time_strip_ind+64]

    for g in range(2):
        for h in range(2):
            print("g and h", g, h)
            temporal_strip_list = []
            for filepath in considered_filepaths:
                numpy_array = np.load(filepath)
                new_slice = np.expand_dims(numpy_array[5000*g:5000*(g+1), 5000*h:5000*(h+1)], axis=0)
                temporal_strip_list.append(new_slice)
            
            temporal_strip = np.concatenate(temporal_strip_list, axis=0)

            labels_ = labels[5000*g:5000*g+5000, 5000*h:5000*h+5000]
            x_inds, y_inds = np.where(labels_==chosen_crop_types)


            if(len(x_inds)!=0 and len(y_inds)!=0):
                for i in range(intra_loop_length):
                    random_corner = random.choices(range(len(x_inds)-2), k=1)[0]
                    x_corner = x_inds[random_corner]
                    y_corner = y_inds[random_corner]

                    if(x_corner>labels_.shape[0]-90):
                        x_corner = x_corner-90
                    if(y_corner>labels_.shape[1]-90):
                        y_corner = y_corner-90

                    space_label = labels_[x_corner:x_corner+64, y_corner:y_corner+64]
                    temporal_strip_ = temporal_strip[:, x_corner:x_corner+64, y_corner:y_corner+64]
                    new_label = np.expand_dims(space_label, axis=0)
                    spatial_label_batches_list.append(new_label)

                    new_entry = np.expand_dims(temporal_strip_, axis=0)
                    temporal_batches_list.append(new_entry)

                spatial_label_batches = np.concatenate(spatial_label_batches_list, axis=0)
                temporal_batches = np.concatenate(temporal_batches_list, axis=0)
            else:    
                temporal_strip = np.array([])


    all_lai_list.append(temporal_batches)
    all_labels_list.append(spatial_label_batches)


all_lai = np.concatenate(all_lai_list, axis=0)
all_labels = np.concatenate(all_labels_list, axis=0)


print("all_lai.shape", all_lai.shape)
print("all_labels.shape", all_labels.shape)


tifffile.imwrite('./storage/per_crop_data_labels_test/'+vista_crop_dict[chosen_crop_types]+'/train'+vista_crop_dict[chosen_crop_types]+'n.tif', all_lai)
tifffile.imwrite('./storage/per_crop_data_labels_test/'+vista_crop_dict[chosen_crop_types]+'/lab'+vista_crop_dict[chosen_crop_types]+'n.tif', all_labels)

