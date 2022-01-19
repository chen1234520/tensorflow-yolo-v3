# tensorflow-yolo-v3

说明：这个项目是将darknet框架的yolov3 yolov3-tiny模型转换为tensorflow框架。TensorFlow的版本为1.11.0

另外提供darknet框架官方yolo项目：https://github.com/AlexeyAB/darknet#yolo-v4-in-other-frameworks

tensorflow2.x版本的yolo项目：https://github.com/hunglc007/tensorflow-yolov4-tflite


项目亮点：

1.老版本的tensorflow，正好满足部署到特殊NPU的需求（使用的NPU芯片貌似不支持tf2）

2.模型转换时支持check和pb两种格式的tf模型

Implementation of YOLO v3 object detector in Tensorflow (TF-Slim). Full tutorial can be found [here](https://medium.com/@pawekapica_31302/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe).

Tested on Python 3.5, Tensorflow 1.11.0 on Ubuntu 16.04.

## Todo list:
- [x] YOLO v3 architecture
- [x] Basic working demo
- [x] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [ ] Training pipeline
- [ ] More backends

## How to run the demo:
To run demo type this in the command line:

1. Download COCO class names file: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download and convert model weights:    
    1. Download binary file with desired weights: 
        1. Full weights: `wget https://pjreddie.com/media/files/yolov3.weights`
        1. Tiny weights: `wget https://pjreddie.com/media/files/yolov3-tiny.weights` 
        1. SPP weights: `wget https://pjreddie.com/media/files/yolov3-spp.weights` 
    2. Run `python ./convert_weights.py` and `python ./convert_weights_pb.py`        
3. Run `python ./demo.py --input_img <path-to-image> --output_img <name-of-output-image> --frozen_model <path-to-frozen-model>`


####Optional Flags
1. convert_weights: 该脚本是把darknet框架模型转换为tensorflow框架 check格式模型
    1. `--class_names`
        1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--tiny`
        1. Use yolov3-tiny
    5. `--spp`
        1. Use yolov3-spp
    6. `--ckpt_file`
        1. Output checkpoint file
2. convert_weights_pb.py: 该脚本是把darknet框架模型转换为tensorflow框架 pb格式模型(加载模型时不需要网络结构)
    1. `--class_names`
            1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file    
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--tiny`
        1. Use yolov3-tiny
    5. `--spp`
        1. Use yolov3-spp
    6. `--output_graph`
        1. Location to write the output .pb graph to
3. demo.py
    1. `--class_names`
        1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--ckpt_file`
        1. Path to the checkpoint file
    5. `--frozen_model`
        1. Path to the frozen model
    6. `--conf_threshold`
        1. Desired confidence threshold
    7. `--iou_threshold`
        1. Desired iou threshold
    8. `--gpu_memory_fraction`
        1. Fraction of gpu memory to work with

4. demo_list.py

    新增的批量测试脚本，与上面的脚本进行了以下几点修改：
    
    1）图像resize的方式采用直接resize操作，原作者使用填充宽高保持宽比例不变，这种方式与darknet训练时候不一致.
    
    2）相对应的，在显示检测框时也同样修改传入参数，true——> false
    
    注意：还需要注意的是，预处理代码需要跟训练阶段保持一致(是否有归一化等操作)
