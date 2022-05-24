import os
import sys

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np

from PyQt5 import QtGui,QtWidgets,QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import defectdetection
import object_detection

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
#CUSTOM_MODEL_NAME = 'my_efficientdet_d0'
PRETRAINED_MODEL_NAME = 'efficientdet_d0_coco17_tpu-32'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}




# 新建了Example类，借由操作Example来操作export里的Ui_MainWindow对象，
# 这样做的目的是将业务逻辑和函数绑定相关工作全部交给Example，
# 将UI、程序入口、业务逻辑完全分离，方便拓展，这样也符合OOP思想。

class Example(QMainWindow):
    def __init__(self):
        self.app = QApplication(sys.argv)
        super().__init__()
        self.ui = defectdetection.Ui_MainWindow()
        self.ui.setupUi(self)
        # 初始化
        self.init_ui()

    #把opencv格式的图像转换成qt格式
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        #return QPixmap.fromImage(convert_to_Qt_format)
        p = convert_to_Qt_format.scaled(200,200,Qt.AspectRatioMode.KeepAspectRatio)
        #p = convert_to_Qt_format.scaled()
        return QPixmap.fromImage(p)


    #左按钮的响应
    def click_left(self):
        #设置待检测图像
        #self.ui.left_label.setText("haole")
        img_name,img_type = QFileDialog.getOpenFileName(None, "选择图片文件", os.getcwd(), "所有文件 (*.*),*.*")
        img = cv2.imread(img_name)
        qt_img_left=self.convert_cv_qt(img)
        self.ui.img_left.setPixmap(qt_img_left)

        #检测
        #创建labelmap
        labels = [{'name': 'crazing', 'id': 1}, {'name': 'inclusion', 'id': 2}, {'name': 'patches', 'id': 3},
                  {'name': 'pitted_surface', 'id': 4}, {'name': 'rolled-in_scale', 'id': 5},
                  {'name': 'scratches', 'id': 6}]

        with open(files['LABELMAP'], 'w') as f:
            for label in labels:
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')

        #加载pipeline并建立模型
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        #恢复checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

        @tf.function
        def detect_fn(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections

        #检测加载出的图像
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

        tf.config.experimental_run_functions_eagerly(True)#不加这个要出事……

        image_np = np.array(img)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.4,
            agnostic_mode=False)

        qt_img_right=self.convert_cv_qt(image_np_with_detections)
        self.ui.img_right.setPixmap(qt_img_right)
        #self.ui.right_label.setText("haole")


    # ui初始化
    def init_ui(self):
        # 初始化方法
        self.ui.button_left.clicked.connect(self.click_left)

        self.show()


# 程序入口
if __name__ == '__main__':
    e = Example()
    sys.exit(e.app.exec())