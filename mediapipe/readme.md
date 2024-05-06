Use coco json format 


# folder structure

                /dataset
                |---/train
                |---/---/images
                |---/---/labels.json
                |---/valid
                |---/---/images
                |---/---/labels.json

# labels.json

                {
                    "images":[{"id":0,"file_name":"0.jpg"}, ...],
                    "annotations":[{"image_id":0,"bbox":[196,151,27,35],"category_id":1}, ...],
                    "categories":[{"id":1,"name":"B"},...]
                }

