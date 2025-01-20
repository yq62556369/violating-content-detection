import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import MinMaxScaler
import re


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

data = pd.read_excel('basic_data.xlsx', engine='openpyxl')

def preprocess_publish_time(time_str):
    cleaned = re.sub(r'[-:\s]', '', str(time_str))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

data['发布时间_processed'] = data['发布时间'].apply(preprocess_publish_time)

sentiment_mapping_1 = {
    "负面": [1, 0, 0],
    "中性": [0, 1, 0],
    "正面": [0, 0, 1]
}

data['sentiment1_neg'], data['sentiment1_neu'], data['sentiment1_pos'] = zip(*data['情感分类1'].map(sentiment_mapping_1))

sentiment_mapping_2 = {
    "star 1": [1, 0, 0, 0, 0],
    "star 2": [0, 1, 0, 0, 0],
    "star 3": [0, 0, 1, 0, 0],
    "star 4": [0, 0, 0, 1, 0],
    "star 5": [0, 0, 0, 0, 1]
}

sentiment2_encoded = data['情感分类2'].map(sentiment_mapping_2)
for i in range(1, 6):
    data[f'sentiment2_star{i}'] = sentiment2_encoded.apply(lambda x: x[i - 1] if isinstance(x, list) else 0)

label_mapping = {
    'NVP': 0,
    'OP': 1,
    'PVP': 2,
    'OVP': 3
}
data['label'] = data['违规情况'].map(label_mapping)

scaler = MinMaxScaler()
data[['发布时间_processed_normalized', '回复贴_normalized']] = scaler.fit_transform(
    data[['发布时间_processed', '回复贴']]
)

tokenizer = BertTokenizer.from_pretrained('./models/bert-base-chinese')

class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.publish_time = self.df['发布时间_processed_normalized'].values
        self.replies = self.df['回复贴_normalized'].values

        self.sentiment1 = self.df[['sentiment1_neg', 'sentiment1_neu', 'sentiment1_pos']].values
        self.sentiment2 = self.df[[f'sentiment2_star{i}' for i in range(1, 6)]].values

        self.labels = self.df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        bar_name = str(self.df.loc[idx, '所在吧名'])
        encoding_base = self.tokenizer.encode_plus(
            bar_name,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        base_numeric = torch.tensor([
            self.publish_time[idx],
            self.replies[idx]
        ], dtype=torch.float)

        title = str(self.df.loc[idx, '标题'])
        content = str(self.df.loc[idx, '正文文本'])
        reply1 = str(self.df.loc[idx, '回复贴1'])
        reply2 = str(self.df.loc[idx, '回复贴2'])
        concatenated_text = f"{title}\n{content}\n{reply1}\n{reply2}"

        encoding_text = self.tokenizer.encode_plus(
            concatenated_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        sentiment1 = self.sentiment1[idx]
        sentiment2 = self.sentiment2[idx]
        sentiment_features = torch.tensor(
            np.concatenate([sentiment1, sentiment2]),
            dtype=torch.float
        )

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            'input_ids_base': encoding_base['input_ids'].flatten(),
            'attention_mask_base': encoding_base['attention_mask'].flatten(),
            'base_numeric': base_numeric,

            'input_ids_text': encoding_text['input_ids'].flatten(),
            'attention_mask_text': encoding_text['attention_mask'].flatten(),

            'sentiment_features': sentiment_features,

            'labels': label
        }

train_df, temp_df = train_test_split(data, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=2 / 3, random_state=42)

train_dataset = TextClassificationDataset(train_df, tokenizer)
val_dataset = TextClassificationDataset(val_df, tokenizer)
test_dataset = TextClassificationDataset(test_df, tokenizer)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, base_numeric_dim, sentiment_dim, num_classes):
        super(BERTClassifier, self).__init__()

        self.bert_base = BertModel.from_pretrained(bert_model_name)

        self.bert_text = BertModel.from_pretrained(bert_model_name)

        self.dropout = nn.Dropout(0.5)

        in_features = 768 + 768 + base_numeric_dim + sentiment_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(
            self,
            input_ids_base, attention_mask_base,
            base_numeric,
            input_ids_text, attention_mask_text,
            sentiment_features
    ):

        outputs_text = self.bert_text(input_ids=input_ids_text, attention_mask=attention_mask_text)
        cls_text = outputs_text.last_hidden_state[:, 0, :]

        outputs_base = self.bert_base(input_ids=input_ids_base, attention_mask=attention_mask_base)
        cls_base = outputs_base.last_hidden_state[:, 0, :]

        base_feature = torch.cat((cls_base, base_numeric), dim=1)

        combined = torch.cat((cls_text, base_feature, sentiment_features), dim=1)

        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

base_numeric_dim = 2
sentiment_dim = 8
num_classes = 4

model = BERTClassifier(
    bert_model_name='bert-base-chinese',
    base_numeric_dim=base_numeric_dim,
    sentiment_dim=sentiment_dim,
    num_classes=num_classes
)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()

        input_ids_base = batch['input_ids_base'].to(device)
        attention_mask_base = batch['attention_mask_base'].to(device)
        base_numeric = batch['base_numeric'].to(device)

        input_ids_text = batch['input_ids_text'].to(device)
        attention_mask_text = batch['attention_mask_text'].to(device)

        sentiment_features = batch['sentiment_features'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids_base=input_ids_base,
            attention_mask_base=attention_mask_base,
            base_numeric=base_numeric,
            input_ids_text=input_ids_text,
            attention_mask_text=attention_mask_text,
            sentiment_features=sentiment_features
        )

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def eval_model(model, data_loader, criterion, device, report=False):
    model.eval()
    total_loss = 0
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids_base = batch['input_ids_base'].to(device)
            attention_mask_base = batch['attention_mask_base'].to(device)
            base_numeric = batch['base_numeric'].to(device)

            input_ids_text = batch['input_ids_text'].to(device)
            attention_mask_text = batch['attention_mask_text'].to(device)

            sentiment_features = batch['sentiment_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids_base=input_ids_base,
                attention_mask_base=attention_mask_base,
                base_numeric=base_numeric,
                input_ids_text=input_ids_text,
                attention_mask_text=attention_mask_text,
                sentiment_features=sentiment_features
            )

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, batch_preds = torch.max(outputs, dim=1)
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, preds, average='weighted', zero_division=0
    )

    print(
        f"损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}")

    if report:
        print("\nPer-Class Metrics:")

        report_dict = classification_report(
            true_labels, preds,
            target_names=list(label_mapping.values()),
            output_dict=True,
            zero_division=0
        )
        for i, label_name in label_mapping.items():
            class_metrics = report_dict[label_name]
            print(f"Label {i} - Precision: {class_metrics['precision']:.4f}, "
                  f"Recall: {class_metrics['recall']:.4f}, "
                  f"F1 Score: {class_metrics['f1-score']:.4f}")

    return avg_loss, accuracy, precision, recall, f1

epochs = 3

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"训练损失: {train_loss:.4f}")

    val_loss, val_acc, val_prec, val_rec, val_f1 = eval_model(
        model, val_loader, criterion, device, report=False
    )
    print("-" * 50)

print("测试集评估:")
test_loss, test_acc, test_prec, test_rec, test_f1 = eval_model(
    model, test_loader, criterion, device, report=True
)
