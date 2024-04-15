
# Convert huggingface model to gguf and build docker image to run with CPU

1. download huggingface model you need, mine is: https://huggingface.co/Viet-Mistral/Vistral-7B-Chat

                to folder : "/work/llm/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat"

2. clone llamacpp: git clone https://github.com/ggerganov/llama.cpp.git

Open terminal , go to inside foler llama.cpp

                cd llama.cpp

                pip install -r requirements.txt

                # build for run in cpu

                mkdir build
                cd build
                cmake ..
                cmake --build . --config Release

3. convert model to gguf

convert.py in folder llama.cpp cloned

                python convert.py "/work/llm/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat" --outfile Vistral-7B-Chat.gguf --outtype q8_0

4. build docker image and run

                
                copy build/bin to mistravn/bin (in step 2)

                copy Vistral-7B-Chat.gguf to mistravn/Vistral-7B-Chat.gguf (in step 3)

                docker build -f dockerfile.llamaccp -t llama-vistral7b .

                docker run -d --restart always -p 22222:8880 --name llama-vistral7b_8880 llama-vistral7b

5. run bash 

                /work/llama.cpp/build/bin/server -m '/work/llama.cpp/Vistral-7B-Chat.gguf' -c 2048 --host 0.0.0.0 --port 8880

6. dockerfile.llamacpp

                FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
                USER root
                WORKDIR /app 
                RUN apt-get update &&  apt-get install -y build-essential git libc6
                EXPOSE 8880
                COPY /bin/ /app/bin/
                COPY /Vistral-7B-Chat.gguf /app/Vistral-7B-Chat.gguf
                ENV LC_ALL=C.utf8
                CMD [ "/bin/sh", "-c", "./bin/server -m '/app/Vistral-7B-Chat.gguf' -c 2048 --host 0.0.0.0 --port 8880"]

# Ner_Llm_Gpt

Support Vietnamese 

https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0


# llamma.cpp dockerfile

more usage: https://github.com/badpaybad/llama.cpp.docker 

                    # if manual download just uncomment this , and comment curl
                    COPY llava-v1.5-7b-q4-server.llamafile /app/llava-v1.5-7b-q4-server.llamafile

                    # RUN curl -LO https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-q4-server.llamafile
                    RUN chmod 755 /app/llava-v1.5-7b-q4-server.llamafile


                    #sudo docker build --no-cache -f dockerfile -t docker.io/dunp/llamacpp .
                    #docker run  -d --restart always -p 8080:8080 --name llamacpp_8880 docker.io/dunp/llamacpp


                    #https://github.com/karpathy/llama2.c 

# hugginface git access token 

                    https://huggingface.co/settings/profile

                    https://huggingface.co/settings/tokens 

# vinallamaGPT

                    git clone https://huggingface.co/vilm/vinallama-2.7b     

                    git clone https://oauth:hf_...yourtoken....@huggingface.co/vilm/vinallama-7b  


# phoGPT


                    git clone https://huggingface.co/vinai/PhoGPT-7B5-Instruct
                    git clone https://oauth:hf_...yourtoken....@huggingface.co/vinai/PhoGPT-7B5-Instruct/               


