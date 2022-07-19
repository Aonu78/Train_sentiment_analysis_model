import numpy as np
import pandas as pd
from fast_ml.model_development import train_valid_test_split
from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import datasets



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f1 = datasets.load_metric('f1')
accuracy = datasets.load_metric('accuracy')
precision = datasets.load_metric('precision')
recall = datasets.load_metric('recall')

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, sentences=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        if bool(sentences):
            self.encodings = self.tokenizer(self.sentences,truncation = True,padding = True)
        
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


def compute_metrics(eval_pred):
    metrics_dict = {}
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    metrics_dict.update(f1.compute(predictions = predictions, references = labels, average = 'macro'))
    metrics_dict.update(accuracy.compute(predictions = predictions, references = labels))
    metrics_dict.update(precision.compute(predictions = predictions, references = labels, average = 'macro'))
    metrics_dict.update(recall.compute(predictions = predictions, references = labels, average = 'macro'))
    return metrics_dict


def train_model(dataframe,text,label):
    df = pd.read_csv(dataframe)
    # print(df.head())
    df_reviews = df.loc[:, [text, label]].dropna()
    # df_reviews
    df_reviews[label] = df_reviews[label].apply(lambda x: f'{x}' if x != 1 else f'{x}')
    le = LabelEncoder()
    df_reviews[label] = le.fit_transform(df_reviews[label])
    # df_reviews.head()
    (train_texts, train_labels,val_texts, val_labels,test_texts, test_labels) = train_valid_test_split(df_reviews, target = label, train_size=0.8, valid_size=0.1, test_size=0.1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    train_texts = train_texts[text].to_list()
    train_labels = train_labels.to_list()
    val_texts = val_texts[text].to_list()
    val_labels = val_labels.to_list()
    test_texts = test_texts[text].to_list()
    test_labels = test_labels.to_list()

    train_dataset = DataLoader(train_texts, train_labels)
    val_dataset = DataLoader(val_texts, val_labels)
    test_dataset = DataLoader(test_texts, test_labels)

    print (train_dataset.__getitem__(0))

    id2label = {idx:label for idx, label in enumerate(le.classes_)}
    label2id = {label:idx for idx, label in enumerate(le.classes_)}

    config = AutoConfig.from_pretrained('distilbert-base-uncased',num_labels = 2,   # if we use 5_star, 4_star type rating then we use num_labels = 5,  here 5 is the number of star
                                        id2label = id2label,label2id = label2id)
    model = AutoModelForSequenceClassification.from_config(config)

    training_args = TrainingArguments(output_dir='/working/results',
    num_train_epochs=10,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.05,
    report_to='none',
    evaluation_strategy='steps',
    logging_dir='/working/logs',
    # logging_steps=50, # update is comming in next undating system...
    eval_steps = 50)

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics)


    trainer.train()


    eval_results = trainer.predict(test_dataset)

    print (eval_results.metrics) # to produce results...

    label2id_mapper = model.config.id2label
    proba = softmax(torch.from_numpy(eval_results.predictions))
    pred = [label2id_mapper[i] for i in torch.argmax(proba, dim = -1).numpy()]
    actual = [label2id_mapper[i] for i in eval_results.label_ids]
    # watch it in tabel
    class_report = classification_report(actual, pred, output_dict = True)
    print(pd.DataFrame(class_report))

    trainer.save_model('/working/sentiment_model')
    