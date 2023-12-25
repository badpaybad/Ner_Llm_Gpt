# Ner_Llm_Gpt


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


                    git clone https://huggingface.co/vinai/PhoGPT-7
                    git clone https://oauth:hf_...yourtoken....@huggingface.co/vinai/PhoGPT-7B5-Instruct/               


