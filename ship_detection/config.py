import os
import sys

slim_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slim")
sys.path.append(slim_folder)

LOG_FOLDER = r"E:\CNN\logs\faster_rcnn"

DATA_FOLDER = r"E:\CNN\object_detection\tensorflow"

COCO_WEIGHTS_PATH = r"E:\CNN\models\faster_rcnn\faster_rcnn_resnet50_coco_2018_01_28\model.ckpt"

CONFIG_FILE_PATH = os.path.join(DATA_FOLDER, "faster_rcnn_resnet50_network.config")
CONFIG_FILE_TEMPLATE_PATH = os.path.join(DATA_FOLDER, "faster_rcnn_resnet50_network_template.config")

TRAIN_RECORD_PATH = os.path.join(DATA_FOLDER, "train.record")
EVAL_RECORD_PATH = os.path.join(DATA_FOLDER, "eval.record")
LABEL_MAP_PATH = os.path.join(DATA_FOLDER, "label_map.pbtxt")

SCALE_ARRAYS = [
    # [2**0, 2**(1/3), 2**(2/3)],
    # [0.5, 1.0, 1.5],
    # [0.25, 0.5, 1.0],
    [0.25, 0.5, 1.0, 2.0],
    # [0.125, 0.25, 0.5, 1.0],
    # [0.0625, 0.125, 0.25, 0.5, 1.0]
]

STRIDES = [16] # 8
RUN_NRS = ["1st"] # "2nd", "3rd"

EPOCHS = 20
STEP_SIZE = 735
MAX_STEPS = EPOCHS * STEP_SIZE

INFERENCE_THRESHOLD= 0.7

COCO_GROUND_TRUTH_PATH = os.path.join(DATA_FOLDER, "coco_ships_eval.json")
COCO_CATEGORY_ID = 1
COCO_RESULTS_FILE_NAME = "coco_results.json"


def get_config_file_template():
    config_file_template = open(CONFIG_FILE_TEMPLATE_PATH).read()
    config_file_template = config_file_template.replace("$train_record_path$", TRAIN_RECORD_PATH)
    config_file_template = config_file_template.replace("$eval_record_path$", EVAL_RECORD_PATH)
    config_file_template = config_file_template.replace("$label_map_path$", LABEL_MAP_PATH)
    config_file_template = config_file_template.replace("$coco_weights_path$", COCO_WEIGHTS_PATH)
    config_file_template = config_file_template.replace("\\", "/")
    
    return config_file_template