FROM python:3.8-buster

RUN pip3 install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install transformers[torch]

COPY requirements.txt .
RUN pip3 install --no-deps -r requirements.txt

COPY download_pretrained_model.py .
RUN python download_pretrained_model.py

COPY . /app
WORKDIR /app

ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV BASE_MODEL=/root/.cache/huggingface/hub/sentence-transformers__paraphrase-mpnet-base-v2.a867aefa094c578256b01667f75d841e5b7e0eaf/

ENTRYPOINT ["python", "main.py"]
