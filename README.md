# Salient-Object-Detection
This is tensorflow implementation for cvpr2017 paper "Deeply Supervised Salient Object Detection with Short Connections".  
It has been extended to reliably run in Docker.

## Usage
0. Install [Docker Compose](https://docs.docker.com/compose/install/).
1. Download the [Pretrained Model](https://drive.google.com/file/d/0B6l9O8aWij8fVEIxZjQ4ejRzVmc/view?usp=sharing) and put it under ```./inference/salience_model/``
2. Put the images you want to analyze into ```./input/```
3. Run ```docker-compose up```
4. You will find the output under ```./output/```