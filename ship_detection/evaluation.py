import tensorflow as tf
import os
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from ship_detection import config
import numpy as np
import json
from PIL import Image, ImageDraw


coco_image_ids = {}
coco_image_bboxes = {}
coco_results = []

with open(config.COCO_GROUND_TRUTH_PATH) as jsonfile:
    coco_gt = json.load(jsonfile)
    
    for image in coco_gt["images"]:
        coco_image_ids[image["file_name"]] = image["id"]
        
    for annotation in coco_gt["annotations"]:
        bboxes = coco_image_bboxes.setdefault(annotation["image_id"], []) 
        bboxes.append(annotation["bbox"])

label_map = label_map_util.load_labelmap(config.LABEL_MAP_PATH)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=len(label_map.item), use_display_name=True)
category_index = label_map_util.create_category_index(categories) 
    

def extract_fn(data_record):
    features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/filename': tf.FixedLenFeature([], tf.string),
    }
    sample = tf.parse_single_example(data_record, features)
    
    decoded_image = tf.image.decode_image(sample['image/encoded'], channels=3)
    decoded_image = tf.expand_dims(decoded_image, 0)
    
    return decoded_image, sample['image/filename']

def run_inference_for_single_image(graph, session, expanded_image, all_tensor_names):
    tensor_dict = {}
    for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = graph.get_tensor_by_name(
                    tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, expanded_image.shape[1], expanded_image.shape[2])
        detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')

    # Run inference
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    output_dict = session.run(tensor_dict, feed_dict={image_tensor: expanded_image }) # np.expand_dims(image_np, 0)

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def draw_bounding_box(image_draw, bounding_box, color):
    min_x, min_y, max_x, max_y = bounding_box
     
    for stroke_width in range(0, 3):
        image_draw.rectangle([min_x - stroke_width, min_y - stroke_width, 
                              max_x + stroke_width, max_y + stroke_width], outline=color)
        

def visualize_detections(image_pil, bounding_boxes, color):
    image_draw = ImageDraw.Draw(image_pil)

    bounding_boxes_array = []

    for bounding_box in bounding_boxes:
        draw_bounding_box(image_draw, bounding_box, color)
    
    return image_pil, bounding_boxes_array

def infer(folder_name):
    current_log_folder = os.path.join(config.LOG_FOLDER, folder_name)
    inference_model_path = os.path.join(current_log_folder, "inference", "frozen_inference_graph.pb")
    
    if (not os.path.exists(inference_model_path)):
        return
    
    image_output_folder = os.path.join(current_log_folder, "images")
    coco_results_path = os.path.join(current_log_folder, config.COCO_RESULTS_FILE_NAME)
    
    
    if (not os.path.exists(image_output_folder)):
        os.mkdir(image_output_folder)
    
    image_graph = tf.Graph()
    with image_graph.as_default():
        dataset = tf.data.TFRecordDataset([config.EVAL_RECORD_PATH])
        dataset = dataset.map(extract_fn)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
    
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    image_session = tf.Session(graph = image_graph)
    detection_session = tf.Session(graph = detection_graph)
    
    ops = detection_graph.get_operations()        
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    
    try :
        while True:
            decoded_image, image_path = image_session.run(next_element)
            decoded_image_path = image_path.decode()
    
            print(decoded_image_path)
            
            image = Image.fromarray(decoded_image[0])
            gt_image = Image.fromarray(decoded_image[0])
        
            image_width, image_height = image.size               
            
            image_name = os.path.basename(decoded_image_path)
            image_id = coco_image_ids[image_name]
            
            output_dict = run_inference_for_single_image(detection_graph, detection_session, decoded_image, all_tensor_names)
        
            detection_bounding_boxes = []
        
            for i in range(0, len(output_dict['detection_scores'])):
                score = output_dict['detection_scores'][i]
                
                if (score == 0):
                    break
                
                box = output_dict['detection_boxes'][i]
                
                box_min_y = image_height * box[0]
                box_min_x = image_width * box[1]
                box_max_y = image_height * box[2]
                box_max_x = image_width * box[3]
    
                box_height = box_max_y - box_min_y
                box_width = box_max_x - box_min_x
                    
                coco_results.append({
                    "image_id": image_id, 
                    "category_id" : config.COCO_CATEGORY_ID, 
                    "bbox" : [box_min_x, box_min_y, box_width, box_height], 
                    "score" : score.item(),
                })
                
                if score > config.INFERENCE_THRESHOLD:
                    detection_bounding_boxes.append([box_min_x, box_min_y, box_max_x, box_max_y])
                
                
            visualize_detections(image, detection_bounding_boxes, (0,255,255,0))
                 
                
            gt_boxes = coco_image_bboxes.get(image_id, [])
            gt_bounding_boxes = list(map(lambda box: [box[0], box[1], box[0] + box[2], box[1] + box[3]], gt_boxes))
            
            gt_image, _ = visualize_detections(gt_image, gt_bounding_boxes, (255,0,255,0))
            
            image_name_without_ext = os.path.splitext(image_name)[0]
            
            image.save(os.path.join(image_output_folder, image_name_without_ext + ".png"))
            gt_image.save(os.path.join(image_output_folder, image_name_without_ext + "_gt.png"))
        
    except tf.errors.OutOfRangeError:
        pass
        
    with open(coco_results_path, "w") as jsonfile:
        json.dump(coco_results, jsonfile)
    
    
if __name__ == '__main__':
    folder_names = os.listdir(config.LOG_FOLDER)
    
    for folder_name in folder_names:
        infer(folder_name)