# ram or vram require min 12GB

sudo apt install python3 python3-pip
pip3 install -U fastapi uvicorn imutils python-multipart pydantic easydict jwcrypto unidecode requests
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install -U transformers accelerate bitsandbytes

# convert gufl

https://github.com/ggerganov/llama.cpp/discussions/2948

git clone https://github.com/ggerganov/llama.cpp.git

pip install -r requirements.txt

python convert.py "/work/llm/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat" --outfile Vistral-7B-Chat.gguf --outtype q8_0


mkdir build
cd build
cmake ..
cmake --build . --config Release

mkdir build
cd build
cmake .. -DLLAMA_CUDA=ON
cmake --build . --config Release


usage in folder build/bin

./main -m "/work/llm/llama.cpp/Vistral-7B-Chat.gguf" -n 128


./server -m "/work/llm/llama.cpp/Vistral-7B-Chat.gguf" -c 2048


python "/work/llama.cpp/convert.py" "/work/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat" --outfile Vistral-7B-Chat.gguf --outtype q8_0
/work/llama.cpp/build/bin/server -m "/work/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat.gguf" -c 2048 --host 0.0.0.0 --port 11111

copy build/bin to mistravn/bin
copy Vistral-7B-Chat.gguf to mistravn/Vistral-7B-Chat.gguf


docker run --gpus all -v /path/to/models:/models local/llama.cpp:full-cuda --run -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run --gpus all -v /path/to/models:/models local/llama.cpp:light-cuda -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run --gpus all -v /path/to/models:/models local/llama.cpp:server-cuda -m /models/7B/ggml-model-q4_0.gguf --port 8000 --host 0.0.0.0 -n 512 --n-gpu-layers 1


# run nvidia

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

sudo apt-get install -y nvidia-container-toolkit

https://pytorch.org/get-started/locally/

pip3 install torch torchvision torchaudio

                python3 main.py 8080 cuda

# run cpu

                cd mistralvn
                python3 main.py 8080 cpu


# See prerequisites. Adding current user to Video and Render groups AMD

sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/6.0.3/ubuntu/jammy/amdgpu-install_6.0.60003-1_all.deb
wget https://repo.radeon.com/amdgpu-install/6.0.3/ubuntu/focal/amdgpu-install_6.0.60003-1_all.deb
sudo apt install ./amdgpu-install_6.0.60003-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms,rocm
sudo apt install rocm
sudo apt install hiplibsdk
echo "Please reboot system for all settings to take effect."

https://repo.radeon.com/amdgpu-install/6.0.3/ubuntu/jammy/amdgpu-install_6.0.60003-1_all.deb

sudo apt install ./amdgpu-install_6.0.60003-1_all.deb

sudo amdgpu-install --usecase=hiplibsdk,rocmdev,dkms,mllib,mlsdk 

https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/prerequisites.html

sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"

# LLM model


                https://oauth:...your token..@huggingface.co/Viet-Mistral/Vistral-7B-Chat

# XNDA amd npu

                https://github.com/amd/xdna-driver

                git clone -b iommu_sva_part4_v6_v6.8_rc2 https://github.com/AMDESE/linux.git