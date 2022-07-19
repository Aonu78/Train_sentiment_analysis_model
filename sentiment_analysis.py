import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
from torch.nn.functional import softmax

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, sentences=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        if bool(sentences):
            self.encodings = self.tokenizer(self.sentences,
                                            truncation = True,
                                            padding = True)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        if self.labels == None:
            item['labels'] = None
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.sentences)
    
    
    def encode(self, x):
        return self.tokenizer(x, return_tensors = 'pt').to(DEVICE)

class SentimentModel():
    
    def __init__(self, model_path):
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        args =  TrainingArguments(output_dir='/working/results', per_device_eval_batch_size=64)
        self.batch_model = Trainer(model = self.model, args= args)
        self.single_dataloader = DataLoader()
        
    def batch_predict_proba(self, x):
        
        predictions = self.batch_model.predict(DataLoader(x))
        logits = torch.from_numpy(predictions.predictions)
        
        if DEVICE == 'cpu':
            proba = torch.nn.functional.softmax(logits, dim = 1).detach().numpy()
        else:
            proba = torch.nn.functional.softmax(logits, dim = 1).to('cpu').detach().numpy()
        return proba
        
        
    def predict_proba(self, x):
        
        x = self.single_dataloader.encode(x).to(DEVICE)
        predictions = self.model(**x)
        logits = predictions.logits
        
        if DEVICE == 'cpu':
            proba = torch.nn.functional.softmax(logits, dim = 1).detach().numpy()
        else:
            proba = torch.nn.functional.softmax(logits, dim = 1).to('cpu').detach().numpy()
        return proba

def senti_test(single_sentence):
    
    sentiment_model = SentimentModel('/working/sentiment_model')

    single_sentence_probas = sentiment_model.predict_proba(single_sentence)
    id2label = sentiment_model.model.config.id2label
    predicted_class_label = id2label[np.argmax(single_sentence_probas)]
    return predicted_class_label