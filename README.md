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


## Running the tool in the docekr

### Extracting labels from Eurocrops


Load the Docker Image

<pre>
cd stelar_3dunet
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker load -i docker_fin_3dunet.tar
<pre>


Extract Labels from EuroCrops

<pre>
docker run -it -v /home/luser/stelar_3dunet:/app/ docker_fin_3dunet python label_extraction_docker.py
<pre>