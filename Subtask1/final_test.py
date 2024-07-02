import torch
from transformers import DebertaForSequenceClassification, DebertaTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from ordered_set import OrderedSet
from torch import nn
import torch
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
import json
import pdb
import logging.handlers
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from networkx import DiGraph, relabel_nodes, all_pairs_shortest_path_length

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ordered_set import OrderedSet
from peft import LoraConfig, get_peft_model, TaskType
from torch import nn


file_path = 'merged_filetest.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data_train = json.load(file)

# transform data to dataframe
rows = []
possible_labels = OrderedSet(label for entry in data_train for label in entry['labels'])

for entry in data_train:
    id = entry['id']
    text = entry['text']
    labels = entry['labels']
    row = {'id': id, 'text': text}

    for label in possible_labels:
        row[label] = int(label in labels)

    rows.append(row)

df = pd.DataFrame(rows)
data_train = df

# check and replace NaN values
for i in data_train.index:
    if pd.isna(data_train["text"][i]):
        data_train["text"][i] = ""

# validation dataset
file_path = 'dev_subtask1_en.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data_validation = json.load(file)

rows = []
possible_labels = OrderedSet(label for entry in data_validation for label in entry['labels'])

for entry in data_validation:
    id = entry['id']
    text = entry['text']
    labels = entry['labels']
    row = {'id': id, 'text': text}

    for label in possible_labels:
        row[label] = int(label in labels)

    rows.append(row)

df = pd.DataFrame(rows)
data_validation = df

# check and replace NaN Values
for i in data_validation.index:
    if pd.isna(data_validation["text"][i]):
        data_validation["text"][i] = ""

# Initialize tokenizer
deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_encodings_deberta = deberta_tokenizer(list(data_train["text"]), truncation=True, padding=True, return_tensors='pt')
train_encodings_roberta = roberta_tokenizer(list(data_train["text"]), truncation=True, padding=True, return_tensors='pt')

test_encodings_deberta = deberta_tokenizer(list(data_validation['text']), truncation=True, padding=True, return_tensors='pt')
test_encodings_roberta = roberta_tokenizer(list(data_validation['text']), truncation=True, padding=True, return_tensors='pt')


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)


labels = sorted([col for col in data_train.columns if col not in ["text", "id"]])

train_dataset_deberta = CustomDataset(train_encodings_deberta, data_train[labels])
test_dataset_deberta = CustomDataset(test_encodings_deberta, data_validation[labels])

train_dataset_roberta = CustomDataset(train_encodings_roberta, data_train[labels])
test_dataset_roberta = CustomDataset(test_encodings_roberta, data_validation[labels])

train_loader_deberta = DataLoader(train_dataset_deberta, batch_size=4, shuffle=True)
test_loader_deberta = DataLoader(test_dataset_deberta, batch_size=4, shuffle=False)

train_loader_roberta = DataLoader(train_dataset_roberta, batch_size=4, shuffle=True)
test_loader_roberta = DataLoader(test_dataset_roberta, batch_size=4, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Best hyperparameters for each pre-trained model
deberta_model = torch.load(f"testdeberta052_subtask1_epoch_1.pth")
roberta_model = torch.load(f"testroberta21_subtask1_epoch_1.pth")

deberta_model.eval()
roberta_model.eval()

all_deberta_preds = []
all_roberta_preds = []
all_labels = []

with torch.no_grad():
    for deberta_batch, roberta_batch in zip(test_loader_deberta, test_loader_roberta):
        deberta_inputs, deberta_labels = deberta_batch
        roberta_inputs, _ = roberta_batch
        deberta_inputs = {key: val.to(device) for key, val in deberta_inputs.items()}
        roberta_inputs = {key: val.to(device) for key, val in roberta_inputs.items()}
        deberta_labels = deberta_labels.to(device)

        deberta_outputs = torch.sigmoid(deberta_model(**deberta_inputs).logits)
        roberta_outputs = torch.sigmoid(roberta_model(**roberta_inputs).logits)

        all_deberta_preds.extend(deberta_outputs.cpu().numpy())
        all_roberta_preds.extend(roberta_outputs.cpu().numpy())
        all_labels.extend(deberta_labels.cpu().numpy())

binary_deberta_preds = (np.array(all_deberta_preds) > 0.5).astype(int)
binary_roberta_preds = (np.array(all_roberta_preds) > 0.5).astype(int)


classification_report_deberta = classification_report(all_labels, binary_deberta_preds, target_names=data_train.columns[2:])
classification_report_roberta = classification_report(all_labels, binary_roberta_preds, target_names=data_train.columns[2:])

print("DeBERTa Classification Report:\n", classification_report_deberta)
print("RoBERTa Classification Report:\n", classification_report_roberta)

possible_labels = ['Appeal to authority',
 'Appeal to fear/prejudice',
 'Bandwagon',
 'Black-and-white Fallacy/Dictatorship',
 'Causal Oversimplification',
 'Doubt',
 'Exaggeration/Minimisation',
 'Flag-waving',
 'Glittering generalities (Virtue)',
 'Loaded Language',
 "Misrepresentation of Someone's Position (Straw Man)",
 'Name calling/Labeling',
 'Obfuscation, Intentional vagueness, Confusion',
 'Presenting Irrelevant Data (Red Herring)',
 'Reductio ad hitlerum',
 'Repetition',
 'Slogans',
 'Smears',
 'Thought-terminating cliché',
 'Whataboutism']


# Compaire the score
deberta_model.eval()
roberta_model.eval()

all_preds_deberta = []
all_preds_roberta = []

all_labels = []



with torch.no_grad():
    for deberta_batch, roberta_batch in zip(test_loader_deberta, test_loader_roberta):
        deberta_inputs, deberta_labels = deberta_batch
        roberta_inputs, _ = roberta_batch
        deberta_inputs = {key: val.to(device) for key, val in deberta_inputs.items()}
        roberta_inputs = {key: val.to(device) for key, val in roberta_inputs.items()}
        deberta_labels = deberta_labels.to(device)

        deberta_outputs = torch.sigmoid(deberta_model(**deberta_inputs).logits)
        roberta_outputs = torch.sigmoid(roberta_model(**roberta_inputs).logits)



        all_preds_deberta.extend(deberta_outputs.cpu().numpy())
        all_preds_roberta.extend(roberta_outputs.cpu().numpy())
        
        all_labels.append(deberta_labels.cpu().numpy())

binary_preds_deberta = (np.array(all_preds_deberta) > 0.5).astype(int)
binary_preds_roberta = (np.array(all_preds_roberta) > 0.5).astype(int)



output_list_deberta = []
for i, entry in data_validation.iterrows():
    entry_id = entry['id']
    text = entry['text']
    predicted_labels = [label for label, pred in zip(possible_labels, binary_preds_deberta[i]) if pred == 1]
    output_entry = {"id": entry_id, "text": text, "labels": predicted_labels}
    output_list_deberta.append(output_entry)

# Prediction of deberta model
output_file_path = 'predictions_deberta.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_list_deberta, output_file, indent=2, ensure_ascii=False)
    

output_list_roberta = []
for i, entry in data_validation.iterrows():
    entry_id = entry['id']
    text = entry['text']
    predicted_labels = [label for label, pred in zip(possible_labels, binary_preds_roberta[i]) if pred == 1]
    output_entry = {"id": entry_id, "text": text, "labels": predicted_labels}
    output_list_roberta.append(output_entry)

# Prediction of roberta model
output_file_path = 'predictions_roberta.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_list_roberta, output_file, indent=2, ensure_ascii=False)

# Score of roberta model
import subtask_1_2a
_h_precision_score, _h_recall_score, _h_fbeta_score = subtask_1_2a.evaluate_h("predictions_roberta.json", "dev_subtask1_en.json")

print("Precision Score r: ", _h_precision_score)
print("Recall Score: ", _h_recall_score)
print("Fbeta Score: ", _h_fbeta_score)


# Score of deberta model
import subtask_1_2a
_h_precision_score, _h_recall_score, _h_fbeta_score = subtask_1_2a.evaluate_h("predictions_deberta.json", "dev_subtask1_en.json")

print("Precision Score d: ", _h_precision_score)
print("Recall Score: ", _h_recall_score)
print("Fbeta Score: ", _h_fbeta_score)


all_preds = []
all_labels = []

# Model with best score from each model, score_deberta>score_roberta, output label = prediction_deberta, inverse or else
with torch.no_grad():
    for deberta_batch, roberta_batch in zip(test_loader_deberta, test_loader_roberta):
        deberta_inputs, deberta_labels = deberta_batch
        roberta_inputs, _ = roberta_batch
        deberta_inputs = {key: val.to(device) for key, val in deberta_inputs.items()}
        roberta_inputs = {key: val.to(device) for key, val in roberta_inputs.items()}
        deberta_labels = deberta_labels.to(device)

        deberta_outputs = torch.sigmoid(deberta_model(**deberta_inputs).logits)
        roberta_outputs = torch.sigmoid(roberta_model(**roberta_inputs).logits)

        # comparation for each sample
        for i in range(len(deberta_outputs)):
            combined_outputs = []
            for j in range(len(deberta_outputs[i])):
                if deberta_outputs[i][j] > roberta_outputs[i][j]:
                    combined_outputs.append(deberta_outputs[i][j].item())
                else:
                    combined_outputs.append(roberta_outputs[i][j].item())

            all_preds.append(combined_outputs)
            all_labels.append(deberta_labels[i].cpu().numpy())

binary_preds = (np.array(all_preds) > 0.5).astype(int)

classification_report_str = classification_report(all_labels, binary_preds, target_names=data_train.columns[2:])
print("Classification Report:\n", classification_report_str)

output_list = []
for i, entry in data_validation.iterrows():
    entry_id = entry['id']
    text = entry['text']
    predicted_labels = [label for label, pred in zip(possible_labels, binary_preds[i]) if pred == 1]
    output_entry = {"id": entry_id, "text": text, "labels": predicted_labels}
    output_list.append(output_entry)

output_file_path = 'predictions_combined.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_list, output_file, indent=2, ensure_ascii=False)

# 
import subtask_1_2a
_h_precision_score, _h_recall_score, _h_fbeta_score = subtask_1_2a.evaluate_h("predictions_combined.json", "dev_subtask1_en.json")

print("Precision Score c1: ", _h_precision_score)
print("Recall Score: ", _h_recall_score)
print("Fbeta Score: ", _h_fbeta_score)



all_deberta_preds = []
all_roberta_preds = []
all_labels = []
# Model combined deberta and roberta predictions
with torch.no_grad():
    for deberta_batch, roberta_batch in zip(test_loader_deberta, test_loader_roberta):
        deberta_inputs, deberta_labels = deberta_batch
        roberta_inputs, _ = roberta_batch
        deberta_inputs = {key: val.to(device) for key, val in deberta_inputs.items()}
        roberta_inputs = {key: val.to(device) for key, val in roberta_inputs.items()}
        deberta_labels = deberta_labels.to(device)

        deberta_outputs = torch.sigmoid(deberta_model(**deberta_inputs).logits)
        roberta_outputs = torch.sigmoid(roberta_model(**roberta_inputs).logits)

        all_deberta_preds.extend(deberta_outputs.cpu().numpy())
        all_roberta_preds.extend(roberta_outputs.cpu().numpy())
        all_labels.extend(deberta_labels.cpu().numpy())

binary_deberta_preds = (np.array(all_deberta_preds) > 0.5).astype(int)
binary_roberta_preds = (np.array(all_roberta_preds) > 0.5).astype(int)


output_list = []
for i, entry in data_validation.iterrows():
    entry_id = entry['id']
    text = entry['text']
    
    deberta_predicted_labels = [f"{label}" for label, pred in zip(possible_labels, binary_deberta_preds[i]) if pred == 1]
    roberta_predicted_labels = [f"{label}" for label, pred in zip(possible_labels, binary_roberta_preds[i]) if pred == 1]
    # labels combination
    combined_labels = deberta_predicted_labels + roberta_predicted_labels
    
    output_entry = {
        "id": entry_id,
        "text": text,
        "labels": combined_labels
    }
    output_list.append(output_entry)


output_file_path = 'predictions_combined2.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_list, output_file, indent=2, ensure_ascii=False)



import subtask_1_2a
_h_precision_score, _h_recall_score, _h_fbeta_score = subtask_1_2a.evaluate_h("predictions_combined2.json", "dev_subtask1_en.json")

print("Precision Score  c2: ", _h_precision_score)
print("Recall Score: ", _h_recall_score)
print("Fbeta Score: ", _h_fbeta_score)