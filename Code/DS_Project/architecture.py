import tensorflow as tf

Model = tf.keras.Model
Input = tf.keras.Input
Dense = tf.keras.layers.Dense
Concatenate = tf.keras.layers.Concatenate
Attention = tf.keras.layers.Attention
plot_model = tf.keras.utils.plot_model

# Number of classes for 'Intent' (Replace with the actual number of intent classes)
num_intent_classes = 1937  # This should be the number of unique intents in your dataset

# Input layers
input_category = Input(shape=(383,), name="input_category")
input_sub_category = Input(shape=(383,), name="input_sub_category")
input_entity = Input(shape=(383,), name="input_entity")
input_cleaned_description = Input(shape=(383,), name="input_cleaned_description")
input_faq = Input(shape=(383,), name="input_faq")
input_faq_answers = Input(shape=(383,), name="input_faq_answers")

# Concatenate all input layers
concatenated = Concatenate()([
    input_category, 
    input_sub_category, 
    input_entity, 
    input_cleaned_description,
    input_faq,
    input_faq_answers
])

# Contextual Understanding Layer
contextual_layer = Attention(use_scale=True)([concatenated, concatenated])

# Intent Classification Layer
intent_output = Dense(128, activation="relu")(contextual_layer)
intent_output = Dense(64, activation="relu")(intent_output)
intent_output = Dense(num_intent_classes, activation="softmax", name="intent_output")(intent_output)

# Define the model
model = Model(
    inputs=[
        input_category, 
        input_sub_category, 
        input_entity, 
        input_cleaned_description,
        input_faq,
        input_faq_answers
    ], 
    outputs=[intent_output]
)

# Compile the model
model.compile(optimizer='adam', loss={"intent_output": "categorical_crossentropy"}, metrics=["accuracy"])

# Summary of the model architecture
model.summary()

# Optional: Save the model architecture to a file
plot_model(model, to_file='model_architecture.png', show_shapes=True)
