import numpy as np
import pandas as pd
from gensim.models.fasttext import load_facebook_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Concatenate,
    Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import os


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

fasttext_path = './models/cc.zh.300.bin'
fasttext_ft = load_facebook_model(fasttext_path)
fasttext_model = fasttext_ft.wv

data = pd.read_excel('basic_data.xlsx', engine='openpyxl')

def get_average_vector(text, model, vector_dim=300):
    words = str(text).split()
    valid_words = [word for word in words if word in model]
    if not valid_words:
        return np.zeros(vector_dim)
    vectors = [model[word] for word in valid_words]
    return np.mean(vectors, axis=0)

data['所在吧名_vector'] = data['所在吧名'].apply(lambda x: get_average_vector(x, fasttext_model))

data['combined_text'] = data[['标题', '正文文本', '回复贴1', '回复贴2']].astype(str).agg('\n'.join, axis=1)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['combined_text'])
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(data['combined_text'])
max_sequence_length = 512
X_text = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

data['发布时间_processed'] = data['发布时间'].astype(str).str.replace(r'[-:\s]', '', regex=True).astype(float)
scaler_publish = MinMaxScaler()
publish_scaled = scaler_publish.fit_transform(data[['发布时间_processed']])

scaler_reply = MinMaxScaler()
reply_scaled = scaler_reply.fit_transform(data[['回复贴']])

sentiment1_mapping = {'负面': 0, '中性': 1, '正面': 2}
data['情感分类1_encoded'] = data['情感分类1'].map(sentiment1_mapping)
sentiment1_ohe = to_categorical(data['情感分类1_encoded'], num_classes=3)

sentiment2_mapping = {'star 1': 0, 'star 2': 1, 'star 3': 2, 'star 4': 3, 'star 5': 4}
data['情感分类2_encoded'] = data['情感分类2'].map(sentiment2_mapping)
sentiment2_ohe = to_categorical(data['情感分类2_encoded'], num_classes=5)

X_bar = np.vstack(data['所在吧名_vector'].values)
X_publish_reply = np.hstack([publish_scaled, reply_scaled])
X_base = np.hstack([X_bar, X_publish_reply])

X_sentiment = np.hstack([sentiment1_ohe, sentiment2_ohe])

label_mapping = {
    'NVP': 0,
    'OP': 1,
    'PVP': 2,
    'OVP': 3
}
data['label'] = data['违规情况'].map(label_mapping)
y = to_categorical(data['label'], num_classes=4)

X_train_text, X_temp_text, X_train_base, X_temp_base, X_train_sentiment, X_temp_sentiment, y_train, y_temp = train_test_split(
    X_text, X_base, X_sentiment, y, test_size=0.3, random_state=SEED)

X_val_text, X_test_text, X_val_base, X_test_base, X_val_sentiment, X_test_sentiment, y_val, y_test = train_test_split(
    X_temp_text, X_temp_base, X_temp_sentiment, y_temp, test_size=(2 / 3), random_state=SEED)

embedding_dim = 300
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if word in fasttext_model:
        embedding_matrix[i] = fasttext_model[word]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

text_input = Input(shape=(max_sequence_length,), name='text_input')
embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_sequence_length,
    trainable=False
)(text_input)
conv_layer = Conv1D(filters=100, kernel_size=2, strides=1, activation='relu')(embedding_layer)
pool_layer = GlobalMaxPooling1D()(conv_layer)

base_input = Input(shape=(X_base.shape[1],), name='base_input')
base_dense = Dense(256, activation='relu')(base_input)

sentiment_input = Input(shape=(X_sentiment.shape[1],), name='sentiment_input')
sentiment_dense = Dense(256, activation='relu')(sentiment_input)

merged = Concatenate()([pool_layer, base_dense, sentiment_dense])

dense = Dense(256, activation='relu')(merged)
dense = Dropout(0.5)(dense)
output = Dense(4, activation='softmax')(dense)

model = Model(inputs=[text_input, base_input, sentiment_input], outputs=output)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

history = model.fit(
    [X_train_text, X_train_base, X_train_sentiment],
    y_train,
    epochs=10,
    batch_size=128,
    validation_data=([X_val_text, X_val_base, X_val_sentiment], y_val),
    verbose=1
)

y_pred_prob = model.predict([X_test_text, X_test_base, X_test_sentiment])
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1值: {f1:.4f}")

report = classification_report(y_true, y_pred, target_names=label_mapping.keys(), output_dict=True, zero_division=0)
print("Per-Class Metrics:")
for label, metrics in report.items():
    if isinstance(metrics, dict):
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score = metrics['f1-score']
        print(f"Label {label} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
