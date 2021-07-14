FROM python:3.8-slim

RUN apt-get update && apt-get install -y gcc && apt-get clean && \
    pip3 install --no-cache-dir torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
	pip3 install --no-cache-dir transformers[torch]

COPY inference-requirements.txt .
RUN pip3 install --no-cache-dir -r inference-requirements.txt && pip uninstall -y torchvision

COPY download_pretrained_model.py .
RUN python download_pretrained_model.py

WORKDIR app/
COPY model.pt .
COPY class2label.json LICENSE utils.py main.py ./
COPY classification/ ./classification/
COPY clustering ./clustering

ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV BASE_MODEL=/root/.cache/huggingface/hub/sentence-transformers__paraphrase-mpnet-base-v2.a867aefa094c578256b01667f75d841e5b7e0eaf/

ENTRYPOINT ["python", "main.py"]
