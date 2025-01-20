import json
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pandas as pd


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

with open("labels_dict.json", "r") as f:
    labels_dict = json.load(f)
    labels_dict = {int(k): v for k, v in labels_dict.items()}

sub_dicts = {i: [] for i in range(4)}
for node_id, label in labels_dict.items():
    sub_dicts[label].append(node_id)

all_nodes = list(labels_dict.items())
random.shuffle(all_nodes)
nodes, labels = zip(*all_nodes)

train_set, temp_set = train_test_split(nodes, test_size=0.3, random_state=42)
val_set, test_set = train_test_split(temp_set, test_size=2/3, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(HANLayer, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_relations)
        ])
        self.out_features = out_features

    def forward(self, x, adj_matrices):
        out = []
        for i, adj in enumerate(adj_matrices):
            h = torch.mm(adj, x)
            h = self.attention_layers[i](h)
            out.append(h)
        out = torch.stack(out, dim=0)
        out = torch.mean(out, dim=0)
        return out

class HAN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_relations, num_layers=4, dropout_rate=0.5):
        super(HAN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            in_feat = in_features if i == 0 else hidden_features
            out_feat = hidden_features
            self.layers.append(HANLayer(in_feat, out_feat, num_relations))

        self.output_layer = HANLayer(hidden_features, out_features, num_relations)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj_matrices):
        for layer in self.layers:
            x = layer(x, adj_matrices)
            x = torch.relu(x)
            x = self.dropout(x)

        x = self.output_layer(x, adj_matrices)
        return x

def train(model, optimizer, criterion, feature_matrix, adj_matrices, labels, train_idx):
    model.train()
    optimizer.zero_grad()
    output = model(feature_matrix, adj_matrices)
    loss = criterion(output[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, feature_matrix, adj_matrices, labels, idx):
    model.eval()
    with torch.no_grad():
        output = model(feature_matrix, adj_matrices)
        preds = output[idx].argmax(dim=1)

        acc = accuracy_score(labels[idx].cpu(), preds.cpu())
        f1 = f1_score(labels[idx].cpu(), preds.cpu(), average='weighted', zero_division=0)
        precision = precision_score(labels[idx].cpu(), preds.cpu(), average='weighted', zero_division=0)
        recall = recall_score(labels[idx].cpu(), preds.cpu(), average='weighted', zero_division=0)

        report = classification_report(labels[idx].cpu(), preds.cpu(), output_dict=True, zero_division=0)

    return acc, f1, precision, recall, report


class_counts = [len(sub_dicts[i]) for i in range(4)]
class_weights = torch.tensor([1.0 / (count**0.5) for count in class_counts], dtype=torch.float).to(device)

feature_matrix = pd.read_csv("total_features.csv", header=None).values
feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32).to(device)

adj_matrices = []
relationship_types = ["相关用户", "同一贴吧", "同一用户"]
for relation in relationship_types:
    adj_matrix = pd.read_csv(f"{relation}_adjacency_matrix.csv", header=None).values
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    adj_matrices.append(adj_matrix)

labels = torch.tensor([labels_dict[node] for node in range(len(feature_matrix))], dtype=torch.long).to(device)
train_idx = torch.tensor(train_set, dtype=torch.long).to(device)
val_idx = torch.tensor(val_set, dtype=torch.long).to(device)
test_idx = torch.tensor(test_set, dtype=torch.long).to(device)

in_features = feature_matrix.shape[1]
hidden_features = 128
out_features = len(set(labels_dict.values()))
num_relations = len(adj_matrices)
num_epochs = 200
learning_rate = 0.1
patience = 20
num_layers = 3
dropout_rate = 0.5

model = HAN(in_features, hidden_features, out_features, num_relations, num_layers=num_layers, dropout_rate=dropout_rate).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_f1 = 0
patience_counter = 0

for epoch in range(num_epochs):

    loss = train(model, optimizer, criterion, feature_matrix, adj_matrices, labels, train_idx)
    val_acc, val_f1, val_precision, val_recall, val_report = evaluate(model, feature_matrix, adj_matrices, labels, val_idx)
    train_acc, train_f1, train_precision, train_recall, train_report = evaluate(model, feature_matrix, adj_matrices, labels, train_idx)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
          f"Validation Acc: {val_acc:.4f}, Validation F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"验证集 F1 分数未提升，提前停止训练于 Epoch {epoch+1}")
        break

test_acc, test_f1, test_precision, test_recall, test_report = evaluate(model, feature_matrix, adj_matrices, labels, test_idx)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

print("\nPer-Class Metrics:")
for label, metrics in test_report.items():
    if label.isdigit():
        print(f"Label {label} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1-score']:.4f}")
