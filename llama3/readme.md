# huggingface-cli

                git clone https://oauth:..token hf here...@huggingface.co/meta-llama/Meta-Llama-3-8B
                huggingface-cli login
                huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/*" --local-dir Meta-Llama-3-8B

# convert

                python "/work/llama.cpp/convert.py" "/work/Meta-Llama-3-8B/original" --outfile "/work/Ner_Llm_Gpt/llama3/Meta-Llama-3-8B.gpu.gguf" --outtype q8_0