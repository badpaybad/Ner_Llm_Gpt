
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

7. api restfull call 

Keep history chat by newline newline 
\n\nUser: ...prompt...\nLlama: ...response...\n\n...

                POST: http://localhost:22222/completion

request: 

                {"stream":false,"n_predict":400,"temperature":0.7,"stop":["</s>","Llama:","User:"],"repeat_last_n":256,"repeat_penalty":1.18,"penalize_nl":false,"top_k":40,"top_p":0.95,"min_p":0.05,"tfs_z":1,"typical_p":1,"presence_penalty":0,"frequency_penalty":0,"mirostat":0,"mirostat_tau":5,"mirostat_eta":0.1,"grammar":"","n_probs":0,"min_keep":0,"image_data":[],"cache_prompt":true,"api_key":"","prompt":"This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.\n\nUser: hi\nLlama: Hello! How may I help you?\n\n\nUser: giới thiệu về Việt Nam\nLlama:"}

response:

                {"content":" Việt Nam là một quốc gia xinh đẹp ở Đông Nam Á với lịch sử và văn hóa phong phú, nổi tiếng với những cảnh quan thiên nhiên tuyệt vời như Vịnh Hạ Long hay động Phong Nha. Nó cũng có nền ẩm thực đa dạng phản ánh di sản lâu đời của nó ","id_slot":0,"stop":true,"model":"/app/Vistral-7B-Chat.gguf","tokens_predicted":58,"tokens_evaluated":78,"generation_settings":{"n_ctx":2048,"n_predict":-1,"model":"/app/Vistral-7B-Chat.gguf","seed":4294967295,"temperature":0.699999988079071,"dynatemp_range":0.0,"dynatemp_exponent":1.0,"top_k":40,"top_p":0.949999988079071,"min_p":0.05000000074505806,"tfs_z":1.0,"typical_p":1.0,"repeat_last_n":256,"repeat_penalty":1.1799999475479126,"presence_penalty":0.0,"frequency_penalty":0.0,"penalty_prompt_tokens":[],"use_penalty_prompt_tokens":false,"mirostat":0,"mirostat_tau":5.0,"mirostat_eta":0.10000000149011612,"penalize_nl":false,"stop":["</s>","Llama:","User:"],"n_keep":0,"n_discard":0,"ignore_eos":false,"stream":false,"logit_bias":[],"n_probs":0,"min_keep":0,"grammar":"","samplers":["top_k","tfs_z","typical_p","top_p","min_p","temperature"]},"prompt":"This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.\n\nUser: hi\nLlama: Hello! How may I help you?\n\n\nUser: giới thiệu về Việt Nam\nLlama:","truncated":false,"stopped_eos":true,"stopped_word":false,"stopped_limit":false,"stopping_word":"","tokens_cached":135,"timings":{"prompt_n":1,"prompt_ms":196.862,"prompt_per_token_ms":196.862,"prompt_per_second":5.079700500858469,"predicted_n":58,"predicted_ms":11245.507,"predicted_per_token_ms":193.88805172413794,"predicted_per_second":5.157615392529657}}


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


