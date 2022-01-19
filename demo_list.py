# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '', 'Input image')
tf.app.flags.DEFINE_string(
    'input_imgs', '', 'Input images')
tf.app.flags.DEFINE_string(
    'output_img', '', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_bool(
    'spp', False, 'Use SPP version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')

def main(argv=None):
    FLAGS.input_imgs = "/home/mgchen/project/object/tensorflow/data/ssy0103"
    FLAGS.output_img = "./result/yolov3-tiny_head_90000"
    FLAGS.tiny = True
    FLAGS.conf_threshold = 0.3
    FLAGS.data_format = "NHWC" # NCHW (gpu only) / NHWC cpu
    FLAGS.class_names = "head.names"
    # FLAGS.frozen_model 和 FLAGS.ckpt_file 模型二选一
    FLAGS.frozen_model = "weights/yolov3-tiny-2022018/yolov3-tiny_head_90000.pb"    #
    # FLAGS.ckpt_file = "./weights/yolov3-tiny-2022018/yolov3-tiny_head_90000.ckpt"

    if not os.path.exists(FLAGS.output_img):
        os.makedirs(FLAGS.output_img)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    classes = load_coco_names(FLAGS.class_names)    # class_map

    # step1.加载模型(pb模型和check模型有不同的加载方式)
    if FLAGS.frozen_model:  # case1.frozen_model加载方式
        t0 = time.time()
        frozenGraph = load_graph(FLAGS.frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)
        sess = tf.Session(graph=frozenGraph, config=config)

    elif FLAGS.ckpt_file:   # case2.ckpt_file加载方式
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        elif FLAGS.spp:
            model = yolo_v3.yolo_v3_spp
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), FLAGS.size, FLAGS.data_format)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        sess = tf.Session(config=config)
        t0 = time.time()
        saver.restore(sess, FLAGS.ckpt_file)
        print('Model restored in {:.2f}s'.format(time.time() - t0))
    else:
        print("error! no model_file!!!")

    # step2.处理图片
    for filename in os.listdir(FLAGS.input_imgs):
        image_path = FLAGS.input_imgs + "/" + filename
        print(image_path)
        img = Image.open(image_path)
        img_resized = np.asarray(img.resize((FLAGS.size, FLAGS.size)))  # resize方式一，直接resize
        # img_resized = img_resized / 255.0
        # img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)  #resize方式二，保持宽高比例不变填充宽高
        img_resized = img_resized.astype(np.float32)


        t0 = time.time()
        detected_boxes = sess.run(
            boxes, feed_dict={inputs: [img_resized]})

        filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=FLAGS.conf_threshold,
                                             iou_threshold=FLAGS.iou_threshold) # nms
        print("Predictions found in {:.2f}s".format(time.time() - t0))

        # 画出检测框 最后一个参数表示网络输入图像与原始图像尺度是否一致,如果使用了resize需要选择false
        draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), False)

        img.save(FLAGS.output_img + "/" + filename)

    sess.close()    # 关闭sess

if __name__ == '__main__':
    tf.app.run()
