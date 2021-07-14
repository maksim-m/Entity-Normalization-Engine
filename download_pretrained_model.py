import huggingface_hub

base_model = "sentence-transformers/paraphrase-mpnet-base-v2"
revision = "a867aefa094c578256b01667f75d841e5b7e0eaf"

model_path = huggingface_hub.snapshot_download(base_model, revision)
print(model_path)
