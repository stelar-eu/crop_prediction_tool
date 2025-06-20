# crop_prediction_tool

## Dockerization step


### Create a Temporary Directory for Docker Installation

<pre>
mkdir /your/address/docker-tmp
</pre>

### Build the Docker Container

Run the following commands to build and save the Docker container:

<pre>
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
DOCKER_TMPDIR=/your/address/docker-tmp \
DOCKER_CONFIG=$HOME/.docker docker build -t docker_fin_3dunet . && \
docker save -o /docker_file/address_in/crop_prediction_tool/docker_fin_3dunet.tar docker_fin_3dunet
</pre>



## Data processing 

Put the LAi zip file in `/dataset/france2/lai_ras/` . and unzip it. 



## Running the tool in the docekr


Load the Docker image and run the crop prediction application

<pre>
cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet
</pre>


<pre>
python lai_extraction_from_ras.py
</pre>

Extract Labels from EuroCrops. Download the crop labels for the required country from https://zenodo.org/records/8229128 and extract and move the files to `./dataset/shape_files_from_Eurocrops`. In `./lai_extraction_from_ras.py` enter the eastings and northings of the selected vista bounding box. Then to extract LAI from RAS files, run  

<pre>
python label_extraction_docker.py
</pre>


## Running crop predictor to save the output of the whole tile 


For crop grown in between February and August
<pre>
cd crop_prediction_tool
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
docker run -it -v /home/luser/crop_prediction_tool:/app/ docker_fin_3dunet 

python vista_patch_exp0/vista_testing_comp_f1_docker.py --season Feb_Aug
</pre>


For crop grown in between May and August
<pre>
python vista_patch_exp0/vista_testing_comp_f1_docker.py --season May_Aug
</pre>


For crop grown in between June and October
<pre>
python vista_patch_exp0/vista_testing_comp_f1_docker.py --season Jun_Oct
</pre>


For crop grown in between January and August
<pre>
python vista_patch_exp0/vista_testing_comp_f1_docker.py --season Jan_Aug
</pre>



## Comprehensive evaluation of Ensemble performance to get crop-wise IOU, Accuracy, F1 score and confusion matrix 

## Test set preparation to evaluate the ensemble

### For months February to August

<pre>

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 33 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 36 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 41 --season Feb_Aug


python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 34 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 37 --season Feb_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 40 --season Feb_Aug

python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 34 --crop_2 37 --crop_3 40
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 33 --crop_2 36 --crop_3 41

</pre>


### For months May to August


<pre>

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 2 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 15 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 20 --season May_Aug

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 21 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 23 --season May_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 28 --season May_Aug

python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 2 --crop_2 15 --crop_3 20
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 21 --crop_2 23 --crop_3 28

</pre>

### For months June to October

<pre>

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 8 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 30 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 9 --season Jun_Oct

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 18 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 19 --season Jun_Oct
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 7 --season Jun_Oct

python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 8 --crop_2 9 --crop_3 30
python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 7 --crop_2 18 --crop_3 19

</pre>


### For months January to August

<pre>

python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 4 --season Jan_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 7 --season Jan_Aug
python vista_patch_exp0/test_set_sampler.py --chosen_crop_types 9 --season Jan_Aug

python vista_patch_exp0/test_set_subgroup_aggregator.py --crop_1 4 --crop_2 7 --crop_3 9

</pre>



### TO evaluate the model and plot and get quantitative results  

<pre>

python vista_patch_exp0/evaluator.py --season Feb_Aug --num_subgroup_samples 300
python vista_patch_exp0/evaluator.py --season May_Aug --num_subgroup_samples 300
python vista_patch_exp0/evaluator.py --season Jun_Oct --num_subgroup_samples 300
python vista_patch_exp0/evaluator.py --season Jan_Aug --num_subgroup_samples 300

python vista_patch_exp0/metrics_and_visuals.py


</pre>