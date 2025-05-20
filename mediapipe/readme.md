Use coco json format 

# coco format

convert others format to coco format, then use mediapipe to train

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
# pip install


                sudo apt install libgtk2.0-dev pkg-config
                sudo apt-get install -y libxcb-xinerama0 libxcb-xinerama0-dev
                sudo apt-get install qt5-default
                sudo apt-get install libgtk-3-dev

                python3 -m pip install --upgrade pip                
                pip3 install -U tensorflow mediapipe mediapipe_model_maker opencv-python opencv-contrib-python  PyQt5
               
                pip3 install -U opencv-python opencv-contrib-python 
                 pip3 install -U mediapipe
                  pip3 install -U mediapipe_model_maker

# chu nom dir struct

download zip 

                https://l.facebook.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1QZ2q9aSIln5rUo8EFvy_VIn_VdmmFqMy%2Fview%3Fusp%3Ddrive_link%26fbclid%3DIwZXh0bgNhZW0CMTAAAR2FZc7gUEZxAcx3V1vw9yD-ie13yC-xYeiCkEQyHBLfuXs8gf0YT2uOy4U_aem_Ae5i9JOjOnptid_ZB3X8KGc8ZEn0CIx1D9JVuQt06IWuQVdvmuUQYNZudsx6zi03PThO73gcpY2_I28P_iCMzZeN&h=AT1Jirw8TCylk4OE04A9IUiVbRqbmujG4rj_v6KBfg0Gn80fdcrKBjoIBziySTz1-BB0EcIOrHuWjCYxz72f7L6okrim0alkxhH9D7iQUsxuPaIwkV8WEDrXDZhyonwjCdP9La6SqQs

extract

                /work/llm/Ner_Llm_Gpt/mediapipe/train-val


                /train-val/wb_localization_dataset
                |---------/images
                |---------/------/train
                |---------/------/val
                |---------/labels
                |---------/------/train
                |---------/------/val


                python convetchunom2coco.py

                will create coco dataset 

                /work/llm/Ner_Llm_Gpt/mediapipe/chunomdataset 

# chu nom train , predict

                python train.py

after train done: model will save to : /work/llm/Ner_Llm_Gpt/mediapipe/exported_model

                python inference.py


![image](https://github.com/badpaybad/Ner_Llm_Gpt/blob/main/mediapipe/detected.jpg)

