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


sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt install python3.10-venv

sudo apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

deactivate
rm -rf venv
python3.10 -m venv venv

            #### neus ko duoc python3.10 -m venv venv --without-pip
            # Tải script cài đặt
            curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

            # Chạy script bằng Python TRONG VENV (quan trọng)
            /work/Ner_Llm_Gpt/mediapipe/venv/bin/python3.10 get-pip.py

            /work/Ner_Llm_Gpt/mediapipe/venv/bin/python3.10 -m pip --version

source venv/bin/activate

curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10

pip install --upgrade pip setuptools wheel
pip install tensorflow==2.13.1
pip install mediapipe==0.10.9 mediapipe-model-maker==0.2.1.4

pip install tensorboard


# chu nom dir struct

download zip 

                https://l.facebook.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1QZ2q9aSIln5rUo8EFvy_VIn_VdmmFqMy%2Fview%3Fusp%3Ddrive_link%26fbclid%3DIwZXh0bgNhZW0CMTAAAR2FZc7gUEZxAcx3V1vw9yD-ie13yC-xYeiCkEQyHBLfuXs8gf0YT2uOy4U_aem_Ae5i9JOjOnptid_ZB3X8KGc8ZEn0CIx1D9JVuQt06IWuQVdvmuUQYNZudsx6zi03PThO73gcpY2_I28P_iCMzZeN&h=AT1Jirw8TCylk4OE04A9IUiVbRqbmujG4rj_v6KBfg0Gn80fdcrKBjoIBziySTz1-BB0EcIOrHuWjCYxz72f7L6okrim0alkxhH9D7iQUsxuPaIwkV8WEDrXDZhyonwjCdP9La6SqQs

                or

                https://drive.google.com/file/d/1yE-bWnkhgz720B1tbOjraTbhqCR2AZdN/view?usp=sharing

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

if want try , this is trained 
https://drive.google.com/file/d/11a_D5CycKh_ThB9EsDTpYgAMr7-2RL1i/view?usp=sharing

                python inference.py


 tensorboard --logdir=exported_model

![image](https://github.com/badpaybad/Ner_Llm_Gpt/blob/main/mediapipe/detected.jpg)

# ref



pip uninstall -y \
  tensorflow tensorflow-cpu \
  tensorflow-text \
  tf-keras \
  tf-models-official \
  keras \
  optree
pip uninstall -y typing-extensions
pip install typing-extensions>=4.8.0


sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils


python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


                sudo apt install libgtk2.0-dev pkg-config
                sudo apt-get install -y libxcb-xinerama0 libxcb-xinerama0-dev
                sudo apt-get install qt5-default
                sudo apt-get install libgtk-3-dev
                sudo apt-get update
                sudo apt-get install python3-dev libyaml-dev

                python3 -m pip install --upgrade pip setuptools wheel
                sudo apt-get update
                sudo apt-get install libyaml-dev

                python3 -m pip install --upgrade pip    
                pip3 install -U pyyaml            
                pip3 install -U PyQt5
               
                pip3 install -U opencv-python opencv-contrib-python pyyaml 

pip install mediapipe==0.10.9
pip install mediapipe-model-maker==0.2.1.4

pip install tensorflow==2.13.1

python_version==3.10
mediapipe==0.10.9
mediapipe-model-maker==0.2.1.4
tensorflow==2.13.1
PyYAML>=6.0.1