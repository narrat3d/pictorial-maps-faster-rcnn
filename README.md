# Detection of sailing ships on historic maps with Faster-RCNN

## Installation

* Requires [Python 3.6.x](https://www.python.org/downloads/)
* Requires [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-downloads) and corresponding [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
* Download [this project](https://gitlab.ethz.ch/sraimund/pictorial-maps-faster-rcnn/-/archive/master/pictorial-maps-faster-rcnn-master.zip)
* pip install -r requirements.txt


## Training

* Download [training data](https://ikgftp.ethz.ch/?u=hpMc&p=uLZy&path=/pictorial_maps_faster_rcnn_data.zip) and adjust DATA_FOLDER in config.py 
* Set LOG_FOLDER in config.py where intermediate snapshots shall be stored
* Download [trained coco weights](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) for Faster-RCNN and set COCO_WEIGHTS_PATH to the downloaded model in config.py
* Optionally adjust properties like strides (e.g. STRIDES = [16]), number of runs (e.g. RUN_NRS = ["1st"]), scales (e.g. SCALE_ARRAYS = [[0.25, 0.5, 1.0, 2.0]]) in config.py
* Run train_and_eval.py to train and validate the network in alternating epochs


## Evaluation

* Run coco_metrics.py to determine the highest coco values from tensorflow event files 
* Optionally set the convert_model flag in coco_metrics.py to export the best model for inference

* Run evaluation.py to output detected and ground truth bounding boxes with ships on historic map validation images
* evaluation.py also stores the coco results as a proof for the calculations during validation


## Inference

* As the results of RetinaNet were better, please use [this network](https://gitlab.ethz.ch/sraimund/pictorial-maps-retinanet) for detecting ships on your own maps or adapt evaluation.py in this project.


## Sources
* https://github.com/tensorflow/models/tree/master/research/object_detection (Apache License, Copyright by The TensorFlow Authors)
* https://github.com/tensorflow/models/tree/master/research/slim (Apache License, Copyright by The TensorFlow Authors)


## Modifications
* Bug fixes based on GitHub issues so that it actually works...
* Compiled .proto to .py files
* Parametrization of config files so that it can be trained in multiples runs with different configurations