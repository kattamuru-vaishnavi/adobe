from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
AutoTokenizer.from_pretrained(model_name).save_pretrained("distiluse-model")
AutoModel.from_pretrained(model_name).save_pretrained("distiluse-model")
