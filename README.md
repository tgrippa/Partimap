# Introduction  
This repository contains the code for predicting perceived deprivation levels from RS image using DL. 
This research was conducted by [Université Libre de Bruxelles](https://anageo.sciences.ulb.be/), in the context of the PARTIMAP project founded by the Belgian science policy (contract SR/11/217).

## Content of this repository
Notebooks/ contains the Jupyter notebooks used for this research.
SRC/ contains custom Python class and functions. 
Dockerfile and singularity.def are image definitions for [Docker](https://www.docker.com/?utm_source=google&utm_medium=cpc&utm_campaign=search_emea_brand&utm_term=docker_exact&gclid=Cj0KCQjw3IqSBhCoARIsAMBkTb383uT5PcW9-8ynZBTU5C7ZE9TRPTFtiD6oscihRfRegF8_2yB7lAYaAhF0EALw_wcB) and [Singularity](https://sylabs.io/guides/3.5/user-guide/introduction.html) respectively. 

## Disclaimer regarding reproducibility
All experiments have been designed on a host environment running on Ubuntu 20.04 (focal) with a Nvidia RTX 2080s (8Gb). The full training and inference of were conducted on a HPC cluster with two Nvidia RTX 2080-Ti (11Gb each) using Singularity containers. There is a high probability that you will not be able to reproduce the experiments with less than 20Gb of VRAM (GPU's RAM).

Beside the actual computer codes, we provide a Docker image and a Singularity image that allow you to create the exact same environment as ours to run the code. Due to the stochastic behaviour of deep learning models, it is not guarantee that results are 100% replicable. 

## Citing this code
TODO : Zenodo/OSF DOI with all co-authors.

# Setup a Docker environment 

## Requirements
- You need to have Docker installed on the host computer (for most of you, it will be your own computer, but it could be a server machine for example). Ensure that Docker is installed by typing ``` bash docker --version ``` in your terminal. If the version is not displayed, please install Docker according to [the official help](https://docs.docker.com/get-docker/).

- You need to have the *NVIDIA Container Toolkit* installed on the Docker host. If it is not the case, follow the [installation instructions](https://www.tensorflow.org/install/docker).

## Download the dataset
=> TODO: Create a repository on OpenScienceFramework and add instruction/commands for download.  

## Clone this repository
Clone this repository on your computer using the following git command or using the direct download [link](https://github.com/tgrippa/Partimap).
```
# Clone repository
git clone https://github.com/tgrippa/Partimap.git
```

## Build the Docker image
Now it is time to build the Docker image you will use for running the code. Be patient, the building may take a while but it is just needed once. The image will be named "partimap_tf2_gpu".
```
# Enter the Git directory
cd Partimap
# Build the Docker image based on the Dockerfile
docker build -t partimap_tf2_gpu .
```

## Run a new Docker instance based on the partimap_tf2_gpu image
Run a new docker instance based on the partimap_tf2_gpu image. After executing the docker command,
you should have a Jupyter lab server running in the Docker instance. Click the IP address of the service that appears in the terminal and a new tab will appear in your web browser.
**Please adapt the [GITHUB_DIR] and [DATA_DIR] with the path on the Docker host to access the Github directory and the data.**

```
# Run a new Docker container named "partimap"
docker run --rm -p 8888:8888 -v [DATA_DIR]:/home/partimap/PARTIMAP_processing -v [GITHUB_DIR]:/home/partimap/Predict_Derpivation_Perceptions_DL --runtime=nvidia --name partimap partimap_tf2_gpu 

```

## Monitor the GPU usage in another terminal 
Launch a bash shell **on another terminal** and launch nvidia-smi utility to monitor the use of the GPU(s).
```
# Open a bash terminal on the running instance named "partimap"
docker exec -it partimap bash
partimap@CONTENER_ID:~$ watch -n 1 nvidia-smi

```

## Root privilege in the container
If you need a root access to the running instance, use the following command.
```
# If you need to use the bash with root privilege (sudo)
docker exec --user="root" -it partimap bash

```

# Setup a Singularity environment 
Singularity is similar to Docker but is available on most High Performing Computing (HPC) cluster, while Docker is not. 

## Requirements
- You need to have Singularity installed on your host computer and on the HPC cluster. Ensure that Docker is installed by typing ``` bash singularity --version ``` in your terminal. If the version is not displayed, please install Singularity according to [the official help](https://sylabs.io/guides/3.0/user-guide/quick_start.html).

## Download the dataset
=> TODO: Create a repository on OpenScienceFramework and add instruction/commands for download.  

## Clone this repository
Clone this repository on your computer using the following git command or using the direct download [link](https://github.com/tgrippa/Partimap).
```
# Clone repository
git clone https://github.com/tgrippa/Partimap.git
```

## Build the Singularity image file on your computer
The easiest way to build your singularity image is to do it on your own computer. 
```
# Enter the Git directory
cd Partimap
# Build the Singularity image file (.sif) based on the definition file (.def)
sudo singularity build partimap_tf28.sif partimap_tf28.def
```

## Transfer the Singularity image file on the HPC cluster
Please consult the instructions of the HPC cluster to transfer the partimap_tf28.sif file. 

## Build the Singularity image on the HPC
Start by connecting to the HPC using SSH. Please be sure to correctly map port 8888 from the host on your localhost:8888 using the correct SSH option ```ssh -L 8888:localhost:8888 <REMOTE_USER>@<REMOTE_HOST> ```.
Once it is done, navigate to the directory where you have copied the github repository and enter the following command.
```
# Build the Singularity image based on the image file (.sif)
singularity build partimap_tf28 partimap_tf28.sif
```

## Run a new Singularity instance based on the partimap_tf28 image
Run an instance based on the partimap_tf2 image. After executing the Singularity command,
you should have a Jupyter lab server running on the HPC and accessible from your own desktop computer. Click the IP address of the service that appears in the terminal and a new tab will appear in your web browser.

```
# Run a new Singularity instance
singularity run --nv partimap_tf28 

```

