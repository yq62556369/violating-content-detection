from py2neo import Graph, NodeMatcher
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, logging
import torch
import re
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.set_verbosity_error()
tokenizer = BertTokenizer.from_pretrained("./models/bert-base-chinese")
model = BertModel.from_pretrained("./models/bert-base-chinese").to(device)
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

node_types = ["NVP", "OP", "PVP", "OVP"]
relationship_types = ["同一用户", "相关用户", "同一贴吧"]

text_bert_dim = 768
bar_bert_dim = 768
publish_time_dim = 1
reply_count_dim = 1
sentiment_1_dim = 3
sentiment_2_dim = 5
fixed_feature_dim = text_bert_dim + bar_bert_dim + publish_time_dim + reply_count_dim + sentiment_1_dim + sentiment_2_dim

labels_dict = {}

label_mapping = {
    "NVP": 0,
    "OP": 1,
    "PVP": 2,
    "OVP": 3
}

sentiment_mapping_1 = {
    "负面": [1, 0, 0],
    "中性": [0, 1, 0],
    "正面": [0, 0, 1]
}

sentiment_mapping_2 = {
    "star 1": [1, 0, 0, 0, 0],
    "star 2": [0, 1, 0, 0, 0],
    "star 3": [0, 0, 1, 0, 0],
    "star 4": [0, 0, 0, 1, 0],
    "star 5": [0, 0, 0, 0, 1]
}

def encode_text(text):

    if not text.strip():
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def pad_or_truncate(vector, length):

    if len(vector) < length:
        return np.pad(vector, (0, length - len(vector)), 'constant')
    else:
        return vector[:length]

def build_feature_matrix():

    all_nodes = []
    matcher = NodeMatcher(graph)

    for node_type in node_types:
        nodes = list(matcher.match(node_type))
        all_nodes.extend(nodes)

    publish_times = []
    reply_counts = []

    for node in all_nodes:
        if "发布时间" in node and node["发布时间"]:
            publish_time_str = re.sub(r"[-:\s]", "", node["发布时间"])
        else:
            publish_time_str = "000000000000"
        try:
            publish_time = float(publish_time_str)
        except ValueError:
            publish_time = 0.0
        publish_times.append(publish_time)

        if "回复贴" in node and node["回复贴"]:
            try:
                reply_count = float(node["回复贴"])
            except ValueError:
                reply_count = 0.0
        else:
            reply_count = 0.0
        reply_counts.append(reply_count)

    min_publish_time = min(publish_times)
    max_publish_time = max(publish_times)
    min_reply_count = min(reply_counts)
    max_reply_count = max(reply_counts)

    publish_time_range = max_publish_time - min_publish_time if max_publish_time > min_publish_time else 1
    reply_count_range = max_reply_count - min_reply_count if max_reply_count > min_reply_count else 1

    all_feature_vectors = []
    node_index = 0

    for node in all_nodes:
        bar_name = node["所在吧名"] if "所在吧名" in node and node["所在吧名"] else ""
        bar_name = re.sub(r"吧$", "", bar_name)
        bar_features = encode_text(bar_name)

        combined_text = "\n".join([
            node[col] if col in node and node[col] else ""
            for col in ["标题", "正文文本", "回复贴1", "回复贴2"]
        ])
        text_features = encode_text(combined_text)

        publish_time = publish_times[node_index]
        publish_time_normalized = (publish_time - min_publish_time) / publish_time_range

        reply_count = reply_counts[node_index]
        reply_normalized = (reply_count - min_reply_count) / reply_count_range

        base_features = np.concatenate([
            bar_features,
            [publish_time_normalized],
            [reply_normalized]
        ])

        sentiment_1 = node.get("情感分类1")
        sentiment_2 = node.get("情感分类2")

        sentiment_1_encoded = sentiment_mapping_1.get(sentiment_1, [0, 0, 0])
        sentiment_2_encoded = sentiment_mapping_2.get(sentiment_2, [0, 0, 0, 0, 0])
        sentiment_features = np.concatenate([sentiment_1_encoded, sentiment_2_encoded])

        feature_vector = np.concatenate([
            text_features,
            base_features,
            sentiment_features
        ])

        node_labels = list(node.labels)
        if node_labels:
            label = label_mapping.get(node_labels[0], -1)
        else:
            label = -1
        labels_dict[node_index] = label

        padded_vector = pad_or_truncate(feature_vector, fixed_feature_dim)
        all_feature_vectors.append(padded_vector)
        node_index += 1

    return np.vstack(all_feature_vectors), all_nodes

def normalize_adjacency_matrix(adj_matrix):

    degree_vector = np.sum(adj_matrix, axis=1)
    inv_sqrt_degree_vector = np.zeros_like(degree_vector)
    inv_sqrt_degree_vector[degree_vector > 0] = np.power(degree_vector[degree_vector > 0], -0.5)
    degree_matrix = np.diag(inv_sqrt_degree_vector)
    normalized_adj = degree_matrix @ adj_matrix @ degree_matrix
    return normalized_adj

def build_adjacency_matrices(all_nodes):

    adjacency_matrices = {}
    node_count = len(all_nodes)

    node_to_index = {node.identity: idx for idx, node in enumerate(all_nodes)}

    for relation in relationship_types:
        adj_matrix = np.zeros((node_count, node_count))

        for i, source_node in enumerate(all_nodes):
            for rel in graph.relationships.match(nodes=(source_node, None), r_type=relation):
                target_node = rel.end_node
                j = node_to_index.get(target_node.identity)
                if j is not None:
                    adj_matrix[i, j] = 1

        adjacency_matrices[relation] = normalize_adjacency_matrix(adj_matrix)

    return adjacency_matrices


total_feature_matrix, all_nodes = build_feature_matrix()
adjacency_matrices = build_adjacency_matrices(all_nodes)

pd.DataFrame(total_feature_matrix).to_csv("total_features.csv", index=False, header=False)
for relation, matrix in adjacency_matrices.items():
    pd.DataFrame(matrix).to_csv(f"{relation}_adjacency_matrix.csv", index=False, header=False)

with open("labels_dict.json", "w") as f:
    json.dump(labels_dict, f)

print("节点特征矩阵和多邻接矩阵已保存至对应文件。")
