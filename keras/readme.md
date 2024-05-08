# python function convert_2_yolo_dataset

Convert coco dataset format into standard: tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

coco ref:

                https://github.com/badpaybad/Ner_Llm_Gpt/tree/main/mediapipe#readme


                ### folder structure

                                /dataset
                                |-------/train
                                |-------/-----/images
                                |-------/-----/labels.json
                                |-------/valid
                                |-------/-----/images
                                |-------/-----/labels.json

                ### labels.json

                                {
                                    "images":[{"id":0,"file_name":"0.jpg"}, ...],
                                    "annotations":[{"image_id":0,"bbox":[196,151,27,35],"category_id":1}, ...],
                                    "categories":[{"id":1,"name":"B"},...]
                                }


from_tensor_slices

                    classes = [
                        [12, 14, 14, 14],     # 4 classes
                        [1],                  # 1 class
                        [7, 7],               # 2 classes
                    ...]

                    bbox = [
                        [[199.0, 19.0, 390.0, 401.0],[217.0, 15.0, 270.0, 157.0],[393.0, 18.0, 432.0, 162.0],               [1.0, 15.0, 226.0, 276.0], [19.0, 95.0, 458.0, 443.0]],     #image 1 has 4 objects
                        [[52.0, 117.0, 109.0, 177.0]],   #image 2 has 1 object
                        [[88.0, 87.0, 235.0, 322.0],[113.0, 117.0, 218.0, 471.0]],   #image 3 has 2 objects
                    ...]

                    image_paths=[
                        "/.../img1.jpg",
                        "/.../img1.jpg",
                        "/.../img2.jpg",
                    ]