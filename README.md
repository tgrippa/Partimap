# Introduction  

This repository contains the code for predicting percieved deprivation level from RS image using DL.

This research was conducted by [Université Libre de Bruxelles](https://anageo.sciences.ulb.be/), in the context of the PARTIMAP project founded by the Belgian science policy (contract XXX/XXX).

XXXXX
XXXXX
XXXXX

## Disclaimer regarding reproducibility
All experiments have been desinged on a desktop environment running on Ubuntu 20.04 (focal) with a Nvidia RTX 2080s (8Gb).
The training and inference of the different experiments were made on a GPU cluster with two Nvidia RTX 2080-Ti (11Gb each) using Singularity containers. 
It is probable that you will not be able to reproduce our experiments with less that 20Gb of VRAM (GPU's RAM).
Beside the actual computer codes, we provide a Docker image and a Singularity image that allow you to create the exact same environment as ours to run the code.
Due to the stochiastic behaviour of deep learning models, it is not garantee that results are 100% replicable. 

## Content of this repository
XXXXX
XXXXX
XXXXX

# Installation

## Requirements
- You need to have Docker installed on the host computer (for most of you, it will be your own computer, but it could be a server machine for example).Ensure Docker is installed by typing ``` bash docker --version ``` in your terminal. If the version is not displayed, please install Docker according to [the official help](https://docs.docker.com/get-docker/).

- We need to have the *NVIDIA Container Toolkit* installed on the Docker host. If it is not the case, follow the installation instructions [here](https://www.tensorflow.org/install/docker).

### Download the dataset
=> TODO: Create a repository on OpenScienceFramework and add instruction/commands for download.  

### Clone this repository
Clone this repository on your computer using the following git command or using the direct download [link](https://github.com/XXXXXXX).
```
# Clone repository
git clone XXXXXXXX
```

### Build the Docker image
Now it is time to build the Docker image you will use for running the code.
Be patient, the building may take a while but it is just needed once.
The image will be named "partimap_tf2_gpu".
```
# Enter the Git directory
cd XXXXXXXXXXXXXX
# Build the Docker image based on the Dockerfile
docker build -t partimap_tf2_gpu .
```

### Run a new Docker instance (a container) base on our partimap_tf2_gpu image
Run a new docker instance based on the mosquimapimage:grass786 image. After executing the docker command,
you should have a Jupyter lab serveur running in the Docker instance.
Click the IP adress of the service and a new tab will appear in your web browser.

```
# Run a new Docker container named "partimap"
docker run -p 8888:8888 -v /media/tais/data/PARTIMAP_processing:/home/partimap/PARTIMAP_processing -v /media/tais/data/Dropbox/ULB/PARTIMAP/Predict_Derpivation_Perceptions_DL:/home/partimap/Predict_Derpivation_Perceptions_DL --runtime=nvidia --name partimap partimap_tf2_gpu:tf2.8 

```

Launch a bash shell **on another terminal** and launch nvidia-smi utility to monitor the use of the GPU.
```
# Open a bash terminal on the running instance named "partimap"
docker exec -it partimap bash
partimap@CONTENER_ID:~$ watch -n 1 nvidia-smi

```

### Root privilege in the container
If you need a root access to the running instance, use the following command.
```
# If you need to use the bash with root privilege (sudo)
docker exec --user="root" -it partimap bash

```

## General usage of the Jupyter Notebooks
The notebooks are made of different types of cells: titles, text (markdown) and Python code.

## Development environment
This working environment has been tested on Ubuntu 20.04.3 LTS (codename=focal) using Docker 20.10.12.

## Acknowledgment
The authors gratefully thanks the XXXXXXXXXXX founding the MOSQUIMAP project.  

## Citing this code
TODO : Zenodo/OSF DOI with all co-authors.
