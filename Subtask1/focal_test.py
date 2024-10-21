import torch
from transformers import DebertaForSequenceClassification, DebertaTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from ordered_set import OrderedSet
from torch import nn
from sklearn.model_selection import train_test_split
import pandas as pd

# Reading file paths
file_paths = ['merged_filetest.json', 'augmented_output.json']

# Merging data
merged_data = []
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        merged_data.extend(data)

# Writing merged JSON file
with open('merged_file22.json', 'w', encoding='utf-8') as merged_file:
    json.dump(merged_data, merged_file, indent=2, ensure_ascii=False)

# Reading merged JSON file
file_path = 'merged_file22.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data_train = json.load(file)

# Converting data to DataFrame
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

# Checking and replacing NaN values
for i in data_train.index:
    if pd.isna(data_train["text"][i]):
        data_train["text"][i] = ""

# Reading validation data
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

# Checking and replacing NaN values
for i in data_validation.index:
    if pd.isna(data_validation["text"][i]):
        data_validation["text"][i] = ""

# Initializing tokenizer
deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_encodings_deberta = deberta_tokenizer(list(data_train["text"]), truncation=True, padding=True, return_tensors='pt')
train_encodings_roberta = roberta_tokenizer(list(data_train["text"]), truncation=True, padding=True, return_tensors='pt')

test_encodings_deberta = deberta_tokenizer(list(data_validation['text']), truncation=True, padding=True, return_tensors='pt')
test_encodings_roberta = roberta_tokenizer(list(data_validation['text']), truncation=True, padding=True, return_tensors='pt')

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)

# Creating datasets and dataloaders
labels = sorted([col for col in data_train.columns if col not in ["text", "id"]])

train_dataset_deberta = CustomDataset(train_encodings_deberta, data_train[labels])
test_dataset_deberta = CustomDataset(test_encodings_deberta, data_validation[labels])

train_dataset_roberta = CustomDataset(train_encodings_roberta, data_train[labels])
test_dataset_roberta = CustomDataset(test_encodings_roberta, data_validation[labels])

train_loader_deberta = DataLoader(train_dataset_deberta, batch_size=4, shuffle=True)
test_loader_deberta = DataLoader(test_dataset_deberta, batch_size=4, shuffle=False)

train_loader_roberta = DataLoader(train_dataset_roberta, batch_size=4, shuffle=True)
test_loader_roberta = DataLoader(test_dataset_roberta, batch_size=4, shuffle=False)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading models
deberta_model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=len(labels)).to(device)
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(labels)).to(device)

# Defining FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Defining optimizer
deberta_optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=1e-5)
roberta_optimizer = torch.optim.AdamW(roberta_model.parameters(), lr=1e-5)

# Defining loss function
criterionR = FocalLoss(alpha=2, gamma=1)
criterionD = FocalLoss(alpha=0.5, gamma=2)

# Training the model
num_epochs = 5

for epoch in tqdm(range(num_epochs)):
    deberta_model.train()
    roberta_model.train()
    running_loss_deberta = 0.0
    running_loss_roberta = 0.0
    for batch in tqdm(train_loader_deberta):
        deberta_optimizer.zero_grad()
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        outputs = deberta_model(**inputs).logits
        loss = criterionD(outputs, labels)
        loss.backward()
        deberta_optimizer.step()
        running_loss_deberta += loss.item()
        
    torch.save(deberta_model, f"testdeberta052_subtask1_epoch_{epoch}.pth")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss_deberta/len(train_loader_deberta)}")
    

    for batch in tqdm(train_loader_roberta):
        roberta_optimizer.zero_grad()
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        outputs = roberta_model(**inputs).logits
        loss = criterionR(outputs, labels)
        loss.backward()
        roberta_optimizer.step()
        running_loss_roberta += loss.item()
        
    torch.save(roberta_model, f"testroberta21_subtask1_epoch_{epoch}.pth")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss_roberta/len(train_loader_roberta)}")
    

# Evaluating the models and outputting results
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

# Calculating accuracy and classification report
binary_deberta_preds = (np.array(all_deberta_preds) > 0.5).astype(int)
binary_roberta_preds = (np.array(all_roberta_preds) > 0.5).astype(int)

classification_report_deberta = classification_report(all_labels, binary_deberta_preds, target_names=data_train.columns[2:])
classification_report_roberta = classification_report(all_labels, binary_roberta_preds, target_names=data_train.columns[2:])

print("DeBERTa Classification Report:\n", classification_report_deberta)
print("RoBERTa Classification Report:\n", classification_report_roberta)

# Writing prediction results to JSON file
possible_labels = [
 'Appeal to authority',
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
 'Whataboutism'
]

# Evaluating models and comparing scores
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

# Writing prediction results to JSON file
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

# Writing prediction results to JSON file
output_file_path = 'predictions_roberta.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_list_roberta, output_file, indent=2, ensure_ascii=False)
