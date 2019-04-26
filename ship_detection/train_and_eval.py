import os
from multiprocessing import Process
from ship_detection import config
from object_detection.legacy import train, eval


def faster_rcnn_train(folder_name):
    print("Start training %s" % folder_name)

    train.FLAGS.train_dir = os.path.join(config.LOG_FOLDER, folder_name)
    train.FLAGS.pipeline_config_path = config.CONFIG_FILE_PATH
    
    train.main(["--logtostderr"])


def faster_rcnn_eval(folder_name):
    print("Start evaluation %s" % folder_name)

    eval.FLAGS.pipeline_config_path = config.CONFIG_FILE_PATH
    eval.FLAGS.checkpoint_dir = os.path.join(config.LOG_FOLDER, folder_name)
    eval.FLAGS.eval_dir = os.path.join(config.LOG_FOLDER, folder_name, "eval")

    eval.main(["--logtostderr"])

if __name__ == '__main__':
    config_file_template = config.get_config_file_template()
    
    for run_nr in config.RUN_NRS:
        for stride in config.STRIDES:
            for scales in config.SCALE_ARRAYS:
                current_step = 0
                
                while current_step < config.MAX_STEPS:
                    current_step += config.STEP_SIZE
                    scales_underscore = "_".join(map(str, scales))
                    scales_comma = ", ".join(map(str, scales))
                    
                    folder_name = "%s_run_stride%s_%s" % (run_nr, stride, scales_underscore)
                    
                    config_file_content = config_file_template.replace("$stride$", str(stride))
                    config_file_content = config_file_content.replace("$scales$", scales_comma)
                    config_file_content = config_file_content.replace("$steps$", str(current_step))
                    
                    with open(config.CONFIG_FILE_PATH, "w") as config_file:
                        config_file.write(config_file_content)
                    
                    train_process = Process(target=faster_rcnn_train, args=(folder_name, ))
                    train_process.start()
                    train_process.join()
                    
                    eval_process = Process(target=faster_rcnn_eval, args=(folder_name, ))
                    eval_process.start()
                    eval_process.join()