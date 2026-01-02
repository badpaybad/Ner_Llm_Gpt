import keras_cv
# In ra danh sách các backbone hỗ trợ
print(keras_cv.models.YOLOV8Backbone.presets.keys())

"""
Using TensorFlow backend
dict_keys(['yolo_v8_xs_backbone', 'yolo_v8_s_backbone', 'yolo_v8_m_backbone', 'yolo_v8_l_backbone', 'yolo_v8_xl_backbone', 'yolo_v8_xs_backbone_coco', 'yolo_v8_s_backbone_coco', 'yolo_v8_m_backbone_coco', 'yolo_v8_l_backbone_coco', 'yolo_v8_xl_backbone_coco'])
"""