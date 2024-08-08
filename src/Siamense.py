from layers import*
from AMPlify import*
from train_amplify import*
from keras.layers import Concatenate

# Assume you have these datasets
# X_active and X_non_active are your datasets of active and non-active peptides
# X_toxic and X_non_toxic are your datasets of toxic and non-toxic peptides

def create_pairs(X1, X2):
    """
    Create pairs of sequences from two datasets.
    X1 and X2 should be numpy arrays of sequences.
    """
    pairs = []
    labels = []
    
    # Positive pairs (similar)
    for i in range(len(X1)):
        for j in range(i + 1, len(X1)):
            pairs.append([X1[i], X1[j]])
            labels.append(1)
    
    # Negative pairs (dissimilar)
    for i in range(len(X1)):
        for j in range(len(X2)):
            pairs.append([X1[i], X2[j]])
            labels.append(0)
    
    return np.array(pairs), np.array(labels)

def build_base_model_attention():
    """
    Build the model architecture for attention output.
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, 
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name='Dropout_1')(hidden)
    hidden, attention = Attention(return_attention=True, name='Attention')(hidden)
    model = Model(inputs=inputs, outputs=[hidden, attention])
    return model

def build_siamese_model():
    """
    Build the Siamese Network model using the shared attention model.
    """
    # Create the base model
    base_model = build_base_model_attention()

    # Define inputs for the Siamese Network
    input_activity = Input(shape=(MAX_LEN, 20), name='Input_Activity')
    input_toxicity = Input(shape=(MAX_LEN, 20), name='Input_Toxicity')

    # Get the outputs from the base model for both inputs
    hidden_activity, attention_activity = base_model(input_activity)
    hidden_toxicity, attention_toxicity = base_model(input_toxicity)

    # Concatenate the hidden states and attentions from both heads
    concat_hidden = Concatenate(name='Concatenate_Hidden')([hidden_activity, hidden_toxicity])
    concat_attention = Concatenate(name='Concatenate_Attention')([attention_activity, attention_toxicity])

    # Add additional layers before producing the final output
    x = Dense(512, activation='relu', name='Dense_Layer_1')(concat_hidden)
    x = Dropout(0.5, name='Dropout_2')(x)
    x = Dense(256, activation='relu', name='Dense_Layer_2')(x)

    # Output layers for activity and toxicity probabilities
    output_activity = Dense(1, activation='sigmoid', name='Output_Activity')(x)
    output_toxicity = Dense(1, activation='sigmoid', name='Output_Toxicity')(x)

    # Create the Siamese model
    siamese_model = Model(inputs=[input_activity, input_toxicity], outputs=[output_activity, output_toxicity])

    # Compile the model
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return siamese_model

# Example usage:
# siamese_model = build_siamese_model()
# siamese_model.summary()

