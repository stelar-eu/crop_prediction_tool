import json
import sys
import traceback
from utils.mclient import MinioClient
'''from utils.processing import prepare_data
from utils.inference import run_inference
from utils.io_utils import save_geotiff, save_png
from utils.config import transform, crs_code, get_labels_in_color'''
import os
import numpy as np
import string
import random


'''

export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock


docker logs 782d5bb1068e -f

conda deactivate
conda deactivate
cd /home/luser/tool_testing/crop_prediction_tool
conda activate /home/luser/miniforge3/envs/stcon4

python main.py input.json output.json

'''

download_LAI = True
download_mask = True

download_models = True

start_evaluating = True

pre_sampling_Feb_Aug = True
pre_sampling_May_Aug = True
pre_sampling_Jun_Oct = True
pre_sampling_Jan_Aug = True

Feb_Aug_subgroup_aggregation = True
May_Aug_subgroup_aggregation = True
Jun_Oct_subgroup_aggregation = True
Jan_Aug_subgroup_aggregation = True

do_all_evaluations = True



def create_random_txt(filename: str) -> None:

    pool = string.ascii_letters + string.digits + string.punctuation + string.whitespace
    random_text = ''.join(random.choices(pool, k=1000))
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(random_text)

import time

def wait_until_file_ready(filepath, timeout=30):
    start_time = time.time()
    while True:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                with open(filepath, 'rb') as f:
                    f.read(1)  # Try to read a byte
                break
            except Exception:
                pass
        if time.time() - start_time > timeout:
            raise TimeoutError(f"File {filepath} not ready after {timeout} seconds.")
        time.sleep(0.2)


def run(json):
    try:
        ######################## MINIO INIT ########################
        #minio_id = json['minio']['id']
        minio_id = "mcyOg4iR5HftaA5KECyc"
        #minio_key = json['minio']['key']
        minio_key = "hmML90voTbcAqDB8zztqYtQdftnqXgFS9dztQCEO"
        #minio_skey = json['minio']['skey']
        minio_endpoint = json['minio']['endpoint_url']

        mc = MinioClient(minio_endpoint, minio_id, minio_key, secure=False)
        inputs = json['input']
        outputs = json['output']

        params = json['parameters']
        chosen_months = params['months_chosen']
        ulx = params['upper_left_x']
        uly = params['upper_left_y']
        pixel_size = params['pixel_size']
        crs_code = params['crs_code']

        print("params", params)
        print("which season here", chosen_months)
        

        # Download LAI
        act_lai_files = inputs['actual_LAI']          # path to LAI folder in MinIO
        outputs = json['output']
        if download_LAI:
            for ts in range (len(act_lai_files)):
                mc.get_object(s3_path=act_lai_files[ts], local_path='./dataset/france2/processed_lai_npy2/'+act_lai_files[ts][40:-6]+str(ts).zfill(2)+'.tif')
            local_path='./dataset/france2/processed_lai_npy2/'+act_lai_files[ts][40:-6]+str(ts).zfill(2)+'.tif'
            wait_until_file_ready(local_path)
            print("done saving LAI")

        # Download Labels
        if download_mask:
            labels = inputs['spatial_labels']          # path to LAI folder in MinIO
            mc.get_object(s3_path=labels[0], local_path='./storage/full_mast1/vista_labes_aligned.npy')
            local_path='./storage/full_mast1/vista_labes_aligned.npy'
            wait_until_file_ready(local_path)
            print("done saving labels")

        # Download Trained models
        trained_models = inputs['models_in_ensemble']          # path to LAI folder in MinIO
        outputs = json['output']
        if download_models:
            for ts in range (len(trained_models)):
                print("trained_models[ts]", trained_models[ts][37:])
                mc.get_object(s3_path=trained_models[ts], local_path='./checkpoints_f1/'+trained_models[ts][37:]+'')
            local_path='./checkpoints_f1/'+trained_models[ts][37:]+''
            wait_until_file_ready(local_path)
            print("done saving models")

        # these imports should be only here and not at the beginning so that the files saved are considered while identifying the paths
        from vista_patch_exp0.spatial_recon import evaluate_months
        from vista_patch_exp0.eval_pre_sampler import LAI_period_sampler
        from vista_patch_exp0.subgroup_aggregator import subgroup_aggregator
        from vista_patch_exp0.final_evaluator import patch_evaluator 
        from vista_patch_exp0.all_metrics_and_visuals import write_eval_results


        # To save full tail
        if start_evaluating:
            for period in (range(len(chosen_months))):
                evaluate_months(chosen_months[period], ulx, uly, pixel_size, crs_code)
        



        # for quantitative evaluation
        if pre_sampling_Feb_Aug:
            crop_groups = params['Feb_Aug_crop_group']       
            for crop in crop_groups:
                LAI_period_sampler(chosen_months[0], crop)
        if Feb_Aug_subgroup_aggregation:
            subgroup = params['Feb_Aug_subgroup1']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)
            subgroup = params['Feb_Aug_subgroup2']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)





        if pre_sampling_May_Aug:
            crop_groups = params['May_Aug_crop_group']         
            for crop in crop_groups:
                LAI_period_sampler(chosen_months[1], crop)
        if May_Aug_subgroup_aggregation:
            subgroup = params['May_Aug_subgroup1']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)
            subgroup = params['May_Aug_subgroup2']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)




        if pre_sampling_Jun_Oct:
            crop_groups = params['Jun_Oct_crop_group']       
            for crop in crop_groups:
                LAI_period_sampler(chosen_months[2], crop)
        if Jun_Oct_subgroup_aggregation:
            subgroup = params['Jun_Oct_subgroup1']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)
            subgroup = params['Jun_Oct_subgroup2']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2],)





        if pre_sampling_Jan_Aug:
            crop_groups = params['Jan_Aug_crop_group']      
            for crop in crop_groups:
                LAI_period_sampler(chosen_months[3], crop)
        if Jan_Aug_subgroup_aggregation:
            subgroup = params['Jan_Aug_subgroup1']
            subgroup_aggregator(subgroup[0], subgroup[1], subgroup[2])


        if do_all_evaluations:
            num_patches = params['num_eval_patches']
            for period in (range(len(chosen_months))):
                patch_evaluator(chosen_months[period], num_patches)

        write_eval_results()

        mc.put_object(s3_path= outputs['predictions']+'crop_type_confusion_matrix.png', file_path="./evaluation_results/crop_type_confusion_matrix.png")
        mc.put_object(s3_path= outputs['predictions']+'evaluation_report.txt', file_path="./evaluation_results/evaluation_report.txt")
        mc.put_object(s3_path= outputs['predictions']+'exp2_acc_no_cloud_interpol.png', file_path="./evaluation_results/exp2_acc_no_cloud_interpol.png")
        mc.put_object(s3_path= outputs['predictions']+'exp2_f1_no_cloud_interpol.png', file_path="./evaluation_results/exp2_f1_no_cloud_interpol.png")
        mc.put_object(s3_path= outputs['predictions']+'exp2_iou_no_cloud_interpol.png', file_path="./evaluation_results/exp2_iou_no_cloud_interpol.png")
        mc.put_object(s3_path= outputs['predictions']+'aggregated_predicted_Feb_Aug.tif', file_path="./aggregated_predicted_Feb_Aug.tif")
        mc.put_object(s3_path= outputs['predictions']+'aggregated_predicted_May_Aug.tif', file_path="./aggregated_predicted_May_Aug.tif")
        mc.put_object(s3_path= outputs['predictions']+'aggregated_predicted_Jun_Oct.tif', file_path="./aggregated_predicted_Jun_Oct.tif")
        mc.put_object(s3_path= outputs['predictions']+'aggregated_predicted_Jan_Aug.tif', file_path="./aggregated_predicted_Jan_Aug.tif")

        mc.put_object(s3_path= outputs['predictions']+'Feb_Aug_ground_truth_.png', file_path="./vista_patch_exp0/aggregated_plots_f1_gt/Feb_Aug_ground_truth_.png")
        mc.put_object(s3_path= outputs['predictions']+'Feb_Aug_predicted_.png', file_path="./vista_patch_exp0/aggregated_plots_f1_gt/Feb_Aug_predicted_.png")

        mc.put_object(s3_path= outputs['predictions']+'May_Aug_ground_truth_.png', file_path="./vista_patch_exp0/aggregated_plots_f1_gt/May_Aug_ground_truth_.png")
        mc.put_object(s3_path= outputs['predictions']+'May_Aug_predicted_.png', file_path="./vista_patch_exp0/aggregated_plots_f1_gt/May_Aug_predicted_.png")

        mc.put_object(s3_path= outputs['predictions']+'Jun_Oct_ground_truth_.png', file_path="./vista_patch_exp0/aggregated_plots_f1_gt/Jun_Oct_ground_truth_.png")
        mc.put_object(s3_path= outputs['predictions']+'Jun_Oct_predicted_.png', file_path="./vista_patch_exp0/aggregated_plots_f1_gt/Jun_Oct_predicted_.png")

        mc.put_object(s3_path= outputs['predictions']+'Jan_Aug_ground_truth_.png', file_path="./vista_patch_exp0/aggregated_plots_f1_gt/Jan_Aug_ground_truth_.png")
        mc.put_object(s3_path= outputs['predictions']+'Jan_Aug_ground_truth_.png', file_path="./vista_patch_exp0/aggregated_plots_f1_gt/Jan_Aug_ground_truth_.png")


        return None
    
    except Exception as e:
        print(traceback.format_exc())
        return {
            'message': 'An error occurred during data processing.',
            'error': traceback.format_exc(),
            'status': 500
        }
        

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Please provide 2 files.")
    with open(sys.argv[1]) as o:
        j = json.load(o)
    response = run(j)
    with open(sys.argv[2], 'w') as o:
        o.write(json.dumps(response, indent=4))


### "predictions": "s3://vista-bucket/test_directory/random_cards.txt"