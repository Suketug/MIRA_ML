import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define Model Architecture
Model = tf.keras.Model
Input = tf.keras.Input
Dense = tf.keras.layers.Dense
Concatenate = tf.keras.layers.Concatenate
Attention = tf.keras.layers.Attention
to_categorical = tf.keras.utils.to_categorical
Dropout = tf.keras.layers.Dropout
l1 = tf.keras.regularizers.l1
l2 = tf.keras.regularizers.l2
EarlyStopping = tf.keras.callbacks.EarlyStopping

# Data Loading
data_path = "Data/Extracted_Data/Training_Data/"
X_train = pd.read_excel(data_path + "X_train_dataset_reduced.xlsx")
Y_train = pd.read_excel(data_path + "Y_train_dataset_reduced.xlsx")
X_val = pd.read_excel(data_path + "X_val_dataset_reduced.xlsx")
Y_val = pd.read_excel(data_path + "Y_val_dataset_reduced.xlsx")
X_test = pd.read_excel(data_path + "X_test_dataset_reduced.xlsx")
Y_test = pd.read_excel(data_path + "Y_test_dataset_reduced.xlsx")

# Combine Y_train, Y_val, and Y_test to find the unique classes
all_Y = pd.concat([Y_train, Y_val, Y_test])
num_intent_classes = all_Y['Intent_encoded'].nunique()


# Input layers
input_category = Input(shape=(128,), name="Category_embedding")
input_sub_category = Input(shape=(128,), name="Sub_Category_embedding")
input_entity = Input(shape=(128,), name="entities_embedding")
input_cleaned_description = Input(shape=(128,), name="Cleaned_Description_embedding")
input_faq = Input(shape=(128,), name="FAQ_embedding")
input_faq_answers = Input(shape=(128,), name="FAQ Answers_embedding")

# Concatenate all input layers
concatenated = Concatenate()([
    input_category,
    input_sub_category,
    input_entity,
    input_cleaned_description,
    input_faq,
    input_faq_answers
])

# Define the model architecture
attention_layer = Attention()([concatenated, concatenated])
dense_layer = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(attention_layer)
dropout_layer = Dropout(0.5)(dense_layer)
output_layer = Dense(num_intent_classes, activation='softmax')(dropout_layer)

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model = Model(inputs=[input_category, input_sub_category, input_entity, input_cleaned_description,
                      input_faq, input_faq_answers], outputs=output_layer)

# Prepare the input data
input_layer_names = ["Category_embedding", "Sub_Category_embedding", "entities_embedding",
                     "Cleaned_Description_embedding", "FAQ_embedding", "FAQ Answers_embedding"]

def prepare_data(df, input_layer_names):
    prepared_data = []
    for layer_name in input_layer_names:
        layer_cols = [col for col in df.columns if col.startswith(layer_name)]
        prepared_data.append(df[layer_cols].values)
    return prepared_data

X_train_prepared = prepare_data(X_train, input_layer_names)
X_val_prepared = prepare_data(X_val, input_layer_names)
X_test_prepared = prepare_data(X_test, input_layer_names)
print("Shape of X_train_prepared:", [data.shape for data in X_train_prepared])
print("Shape of X_val_prepared:", [data.shape for data in X_val_prepared])
print("Shape of X_test_prepared:", [data.shape for data in X_test_prepared])

unique_classes = Y_train['Intent_encoded'].unique()
print("Unique Classes:", unique_classes)
print("Number of Unique Classes:", len(unique_classes))

# Label Preprocessing: One-hot encoding
Y_train_encoded = to_categorical(Y_train['Intent_encoded'], num_classes=num_intent_classes)
Y_val_encoded = to_categorical(Y_val['Intent_encoded'], num_classes=num_intent_classes)
Y_test_encoded = to_categorical(Y_test['Intent_encoded'], num_classes=num_intent_classes)


# Data Scaling
scaler = StandardScaler()
X_train_scaled = [scaler.fit_transform(data) for data in X_train_prepared]
X_val_scaled = [scaler.transform(data) for data in X_val_prepared]
X_test_scaled = [scaler.transform(data) for data in X_test_prepared]

# Model Training with Early Stopping
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    X_train_scaled,
    Y_train_encoded,
    validation_data=(X_val_scaled, Y_val_encoded),
    epochs=50,  # Increased number of epochs
    batch_size=32,
    callbacks=[early_stop]  # Early stopping
)

# Model Evaluation
evaluation_metrics = model.evaluate(
    x=X_test_scaled,
    y=Y_test_encoded,
    batch_size=32
)

print(f"Test Loss: {evaluation_metrics[0]}")
print(f"Test Accuracy: {evaluation_metrics[1]}")

# Save the entire model to a HDF5 file
model.save('mira_model.h5')
print("Saved model to disk")