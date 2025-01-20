import os
import random
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models.fasttext import load_facebook_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (
     Bidirectional, Concatenate, Dense, Dropout, Input, LSTM, Layer, Embedding
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)

        context = inputs * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def get_average_vector(text, model, vector_dim=300):
    words = str(text).split()
    valid_words = [word for word in words if word in model]
    if not valid_words:
        return np.zeros(vector_dim)
    vectors = [model[word] for word in valid_words]
    return np.mean(vectors, axis=0)

data = pd.read_excel('basic_data.xlsx', engine='openpyxl')

label_mapping = {
    'NVP': 0,
    'OP': 1,
    'PVP': 2,
    'OVP': 3
}
data['label'] = data['违规情况'].map(label_mapping)

sentiment1_mapping = {
    '负面': [1, 0, 0],
    '中性': [0, 1, 0],
    '正面': [0, 0, 1]
}
data['sentiment1'] = data['情感分类1'].map(sentiment1_mapping).tolist()

sentiment2_mapping = {
    'star 1': [1, 0, 0, 0, 0],
    'star 2': [0, 1, 0, 0, 0],
    'star 3': [0, 0, 1, 0, 0],
    'star 4': [0, 0, 0, 1, 0],
    'star 5': [0, 0, 0, 0, 1],
}
data['sentiment2'] = data['情感分类2'].map(sentiment2_mapping).tolist()

def preprocess_time(time_str):
    cleaned_str = re.sub(r'[-:\s]', '', str(time_str))
    try:
        return float(cleaned_str)
    except ValueError:
        return 0.0

data['发布时间_processed'] = data['发布时间'].apply(preprocess_time)

scaler_time = MinMaxScaler()
data['发布时间_numeric'] = scaler_time.fit_transform(data[['发布时间_processed']].astype(float))

scaler_reply = MinMaxScaler()
data['回复贴_numeric'] = scaler_reply.fit_transform(data[['回复贴']].astype(float))

fasttext_path = './models/cc.zh.300.bin'
embedding_dim = 300

fasttext_ft = load_facebook_model(fasttext_path)
fasttext = fasttext_ft.wv

data['所在吧名_vector'] = data['所在吧名'].apply(lambda x: get_average_vector(x, fasttext, vector_dim=embedding_dim))

data['main_text'] = (
    data['标题'].astype(str) + '\n' +
    data['正文文本'].astype(str) + '\n' +
    data['回复贴1'].astype(str) + '\n' +
    data['回复贴2'].astype(str)
)
main_texts = data['main_text'].tolist()

tokenizer_main = Tokenizer()
tokenizer_main.fit_on_texts(main_texts)
word_index_main = tokenizer_main.word_index
vocab_size_main = len(word_index_main) + 1

embedding_matrix_main = np.random.normal(size=(vocab_size_main, embedding_dim))

for word, i in word_index_main.items():
    if word in fasttext:
        embedding_matrix_main[i] = fasttext[word]
    else:
        embedding_matrix_main[i] = np.random.normal(size=(embedding_dim,))

sequences_main = tokenizer_main.texts_to_sequences(main_texts)
max_seq_length_main = 512
X_text = pad_sequences(sequences_main, maxlen=max_seq_length_main, padding='post')

X_bar_vector = np.vstack(data['所在吧名_vector'].values)
X_time = data['发布时间_numeric'].values.reshape(-1, 1)
X_reply = data['回复贴_numeric'].values.reshape(-1, 1)
X_base = np.concatenate([X_bar_vector, X_time, X_reply], axis=1)

sentiment1 = np.array(data['sentiment1'].tolist())
sentiment2 = np.array(data['sentiment2'].tolist())
X_sentiment = np.concatenate([sentiment1, sentiment2], axis=1)

y = data['label'].values
num_classes = 4
y = tf.keras.utils.to_categorical(y, num_classes)

X_train_text, X_temp_text, X_train_base, X_temp_base, X_train_sentiment, X_temp_sentiment, y_train, y_temp = train_test_split(
    X_text, X_base, X_sentiment, y, test_size=0.3, random_state=SEED
)
X_val_text, X_test_text, X_val_base, X_test_base, X_val_sentiment, X_test_sentiment, y_val, y_test = train_test_split(
    X_temp_text, X_temp_base, X_temp_sentiment, y_temp, test_size=2/3, random_state=SEED
)

text_input = Input(shape=(max_seq_length_main,), name='text_input')
embedding_main_layer = Embedding(
    input_dim=vocab_size_main,
    output_dim=embedding_dim,
    weights=[embedding_matrix_main],
    input_length=max_seq_length_main,
    trainable=False
)(text_input)

bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_main_layer)

attention_out = AttentionLayer()(bi_lstm)

text_representation = Dense(64, activation='relu')(attention_out)

base_input = Input(shape=(X_base.shape[1],), name='base_input')
base_dense = Dense(64, activation='relu')(base_input)

sentiment_input = Input(shape=(X_sentiment.shape[1],), name='sentiment_input')
sentiment_dense = Dense(64, activation='relu')(sentiment_input)

merged = Concatenate()([text_representation, base_dense, sentiment_dense])

dense_merged = Dense(64, activation='relu')(merged)
dense_merged = Dropout(0.5)(dense_merged)
output = Dense(num_classes, activation='softmax')(dense_merged)

model = Model(inputs=[text_input, base_input, sentiment_input], outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
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
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted', zero_division=0
)

print(f"\n测试集总体准确率: {accuracy:.4f}")
print(f"测试集总体精确率: {precision:.4f}")
print(f"测试集总体召回率: {recall:.4f}")
print(f"测试集总体F1值: {f1:.4f}")

label_names = {v: k for k, v in label_mapping.items()}
report = classification_report(
    y_true,
    y_pred,
    target_names=[label_names[i] for i in range(num_classes)],
    output_dict=True,
    zero_division=0
)

print("\nPer-Class Metrics:")
for i, label_name in label_names.items():
    precision_i = report[label_name]["precision"]
    recall_i = report[label_name]["recall"]
    f1_i = report[label_name]["f1-score"]
    print(f"Label {i}({label_name}) - Precision: {precision_i:.4f}, Recall: {recall_i:.4f}, F1 Score: {f1_i:.4f}")
