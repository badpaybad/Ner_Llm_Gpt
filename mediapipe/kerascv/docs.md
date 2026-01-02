# venv

/work/Ner_Llm_Gpt/mediapipe/kerascv

khởi tạo môi trường python 3.10 rồi cài tensorflow, keras , opencv, numpy ... thật stable để chạy được ngay 

# data set

Theo chuẩn coco:
Train folder image : /work/Ner_Llm_Gpt/mediapipe/chunomdataset/train/images
Train annotation json: /work/Ner_Llm_Gpt/mediapipe/chunomdataset/train/labels.json

Test folder image : /work/Ner_Llm_Gpt/mediapipe/chunomdataset/valid/images
Test annotation json: /work/Ner_Llm_Gpt/mediapipe/chunomdataset/valid/labels.json


# Tạo model và trainning

file code: /work/Ner_Llm_Gpt/mediapipe/kerascv/train.py  

Dùng kreas load các model trainning object detection mạnh để train

Cho phép trong quá trình training push các loss ra file csv và tensorboard

Cho phép lưu lại model, weight, check point để phục vụ quá trình train tiếp được, các check point dùng để kiểm tra chạy thử 

Các check point , model lưu vào folder exported_model
Logs csv lưu vào folder logs 

# convert model thành .tflite

file code: /work/Ner_Llm_Gpt/mediapipe/kerascv/convert.py   
Convert exported model thành  {tên model}.tflite 

# Load .tflite model và checkpoint

fiel code: /work/Ner_Llm_Gpt/mediapipe/kerascv/inference.py 

Để inference test với image /work/Ner_Llm_Gpt/mediapipe/nlvnpf-0137-01-045.jpg

Vẽ kết quả bbox vào file rồi lưu thành {tên model}.test.result.jpg 
