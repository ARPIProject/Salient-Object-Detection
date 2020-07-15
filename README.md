# Salient-Object-Detection
This is a TensorFlow implementation of the model described in the CVPR2017 paper "Deeply Supervised Salient Object Detection with Short Connections".  
It has been extended to reliably run in Docker.

## Usage
0. Install [Docker Compose](https://docs.docker.com/compose/install/).
1. Download the [Pretrained Model](https://drive.google.com/file/d/0B6l9O8aWij8fVEIxZjQ4ejRzVmc/view?usp=sharing) and put it under ```./model/salience_model/```.
2. Run ```docker-compose up -d```.
4. Go to the [Web UI](http://localhost:8501).
5. Run ```docker-compose down``` to stop the UI.
