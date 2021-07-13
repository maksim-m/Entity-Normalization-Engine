import huggingface_hub

model_path = huggingface_hub.snapshot_download("sentence-transformers/paraphrase-mpnet-base-v2")
print(model_path)
