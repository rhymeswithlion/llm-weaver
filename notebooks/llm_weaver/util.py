from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

def get_model_and_tokenizer(identifier):
    tokenizer = AutoTokenizer.from_pretrained(identifier)
    model = TFAutoModelForSequenceClassification.from_pretrained(identifier, from_pt=True)
    return model, tokenizer