# local-llama3-rag-app
streamlit based local llama3 RAG app


## Set up virtual environment

Create virtual environment with

    python3 -m venv .local_llamma3

Source this environemnt

    source .local_llamma3/bin/activate

Install requirements

    pip3 install streamlit streamlit-extras ollama langchain langchain_community bs4 chromadb pypdf faiss-gpu openai


## Setup ollama service in docker

See official documentation [here](https://hub.docker.com/r/ollama/ollama)

Pull and run the ollama docker container

    docker run -d -e CUDA_VISIBLE_DEVICES=0 -e OLLAMA_DEBUG=1 --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:0.1.34

Run the webUI container to manage models, prompts and settings. Advise is ti run both containers seperately, so ollama service can we run and redistributed elsewhere as-is.

    docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main


Linux users need CUDA to be found by docker so we need the [NVIDIA CUDA container toolkit](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/)

