import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

data = pd.read_csv("data/subtask1/train.csv",index_col="id")
#data = data.drop("id",axis=1)

# Check for nan
for i in data.index:
    if(pd.isna(data["text"][i])):
        print(data["text"][i])
        
for i in data.index:
    if(pd.isna(data["text"][i])):
        data["text"][i] = ""
print("done")
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(list(train_df["text"]), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, return_tensors='pt')


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)
    
    
    
labels = []
for col in data.columns:
    if(col not in ["text","id"]):
        labels.append(col)

labels = sorted(labels)
        
train_dataset = CustomDataset(train_encodings, train_df[labels])
test_dataset = CustomDataset(test_encodings, test_df[labels])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(labels))

device = torch.device('cuda')
print("Using device :",torch.cuda.get_device_name(device))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.BCEWithLogitsLoss()


num_epochs = 3
for epoch in tqdm(range(num_epochs)):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        outputs = model(**inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        outputs = torch.sigmoid(model(**inputs).logits)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert predictions to binary format
threshold = 0.5
binary_preds = (np.array(all_preds) > threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(all_labels, binary_preds)
print(f"Accuracy: {accuracy:.2f}")

classification_report_str = classification_report(all_labels, binary_preds, target_names=train_df.columns[1:])
print("Classification Report:\n", classification_report_str)

torch.save(model,"model1.pt")
