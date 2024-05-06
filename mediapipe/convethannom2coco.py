
sourceHanNomFolder="/work/llm/Ner_Llm_Gpt/mediapipe/train-val"

# class x center y center width height 
#./wb_localization_dataset/labels/val/nlvnpf-0137-01-045.txt
#               0 0.882222 0.871589 0.037778 0.041734
#./wb_localization_dataset/images/val/nlvnpf-0137-01-045.jpg

import os,sys
import shutil, cv2

def copy_files(source_dir, destination_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Iterate through files in source directory
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        # Copy the file
        shutil.copy2(source_file, destination_file)
        print(f"File '{filename}' copied to '{destination_dir}'.")
        
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
        
cocoHanNomFolder="/work/llm/Ner_Llm_Gpt/mediapipe/hannomdataset"
cocoHanNomFolderTrain="/work/llm/Ner_Llm_Gpt/mediapipe/hannomdataset/train"
cocoHanNomFolderValid="/work/llm/Ner_Llm_Gpt/mediapipe/hannomdataset/valid"
cocoHanNomFolderTrainImages="/work/llm/Ner_Llm_Gpt/mediapipe/hannomdataset/train/images"
cocoHanNomFolderValidImages="/work/llm/Ner_Llm_Gpt/mediapipe/hannomdataset/valid/images"
create_directory_if_not_exists(cocoHanNomFolder)
create_directory_if_not_exists(cocoHanNomFolderTrain)
create_directory_if_not_exists(cocoHanNomFolderValid)
create_directory_if_not_exists(cocoHanNomFolderTrainImages)
create_directory_if_not_exists(cocoHanNomFolderValidImages)

copy_files(f"{sourceHanNomFolder}/wb_localization_dataset/images/train",cocoHanNomFolderTrainImages)
copy_files(f"{sourceHanNomFolder}/wb_localization_dataset/images/val",cocoHanNomFolderValidImages)


def get_file_name_without_extension(file_path):
    # Get the base name of the file (without directory)
    file_name = os.path.basename(file_path)
    # Split the file name into name and extension
    name, extension = os.path.splitext(file_name)
    return name


dir_labels_train=f"{sourceHanNomFolder}/wb_localization_dataset/labels/train"

def center_to_corner(x, y, w, h, imgW,imgH):    
    return int(x*imgW), int(y*imgH), int(w*imgW), int(h*imgH),

for filename in os.listdir(dir_labels_train):
        # Check if the path is a file
        file_path=os.path.join(dir_labels_train, filename)
        
        filename0Ext=get_file_name_without_extension(file_path)
        
        if os.path.isfile(file_path):
            img= cv2.imread(f"{cocoHanNomFolderTrainImages}/{filename0Ext}.jpg")
            
            imgh,imgw,imgc= img.shape
            
            with open(file_path, 'r') as file:
            # Read the entire contents of the file
                file_contents = file.read()
                lines = file_contents.splitlines()                
                for l in lines:
                    l=l.strip().replace("  "," ")
                    words = l.split()
                    x= float(words[1])
                    y= float(words[2])
                    w= float(words[3])
                    h= float(words[4])
                   
                    x,y,w,h = center_to_corner(x,y,w,h,imgw,imgh)                   
                    
                    drawed= cv2.rectangle(img, (x,y,x+w,x+h), (0, 255, 0),1)
                                        
                    cv2.imshow("Image with Rectangle", drawed)
                    cv2.waitKey(0)
                    
                    
                    

