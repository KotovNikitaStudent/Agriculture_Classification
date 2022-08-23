# Agriculture_Classification
Classification of agricultural crops on UAV images
## About project
This repository provides codes for training and testing crop classification algorithms on UAV optical images. The following architectures of convolutional neural networks are presented in the repository:
* MobilenetV2
## Installation
<b>NOTE:</b> Installation instructions are provided for Linux. 

Install the virtual environment in the project:

```console
pip3 install -r requirements.txt
```
You can also run the project in [Docker container](https://hub.docker.com/layers/205311502/nikkotov98/kotov_sar/dev/images/sha256-24d34b590033f8fdf8f2a50c8042bd3c8cb3491fe7d43bbdb7e0cd5714751d94?context=repo).
Pull docker image from repository:
```console
docker pull nikkotov98/kotov_sar:dev
```
Create a docker container based on the downloaded image:
```console
docker run -it --gpus all -v /local_data_root/:/docker_data_root/ /local_code_root/:/docker_code_root/ --name container_name nikkotov98/kotov_sar:dev
```
You can also use the official image from Nvidia:
```console
docker pull nvcr.io/nvidia/pytorch:21.06-py3
```
## Running
Start training the corresponding algorithm with the following command (optionally, the output of the terminal is written to the log file):
```console
python3 train_xxx.py &> log_name.log
```
Start testing the corresponding algorithm on the saved scales with the following command:
```console
python3 test.py
```
For prepare dataset for training/testing neural network, your data folder must be introduce in the following view:
`
Datafolder
├── class_name_1
├── class_name_2
├── class_name_3
...
`
Then run `extract_patches.py`. There are you should changed parameters `ROOT_DIR`, `DATASET_DIR`, `PATCHES_DIR`, `PATCH_SIZE`, `STEP` to your parameters:
```console
python3 extract_patches.py
```
