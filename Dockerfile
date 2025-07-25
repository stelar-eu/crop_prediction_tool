# Dockerfile, Image, Container
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    bash \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    curl \
    jq \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app/

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CONDA_ENV=stconf
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Accept Conda Terms of Service and create environment
RUN bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda config --set channel_priority strict && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n $CONDA_ENV python=3.11 && \
    conda activate $CONDA_ENV && \
    conda install -y -c conda-forge cudatoolkit=11.8 && \
    pip install nvidia-cudnn-cu11==8.6.0.163 && \
    conda install -y -c conda-forge tensorflow=2.13 && \
    pip install classification-models-3D==1.0.10 && \
    pip install efficientnet-3D==1.0.2 && \
    pip install segmentation-models-3D==1.0.7 && \
    pip install minio && \
    conda install -y -c conda-forge scikit-learn==1.5.0 && \
    conda install -y -c conda-forge matplotlib && \
    pip install patchify==0.2.3 && \
    conda install -y -c conda-forge scikit-image==0.24.0 && \
    conda install -y -c conda-forge rasterio sqlite gdal pyproj fiona geopandas && \
    conda install -y -c conda-forge opencv && \
    conda clean --all --yes"

RUN chmod +x /app/run.sh

ENTRYPOINT ["/app/run.sh"]