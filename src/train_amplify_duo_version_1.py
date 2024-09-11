#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 3: amplify_duo
This script is for model training, and testing the performance if a test set is specified
"""
import os
import argparse
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
from layers import Attention, MultiHeadAttention
from keras.models import Model
from keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout, Concatenate, Layer
from keras.callbacks import EarlyStopping
from keras.optimizers.legacy import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from keras.backend import int_shape
from keras.losses import BinaryCrossentropy

MAX_LEN = 200 # max length for input sequences


def one_hot_padding(seq_list,padding):
    """
    Generate features for aa sequences [one-hot encoding with zero padding].
    Input: seq_list: list of sequences, 
           padding: padding length, >= max sequence length.
    Output: one-hot encoding of sequences.
    """
    feat_list = []
    one_hot = {}
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    for i in range(len(aa)):
        one_hot[aa[i]] = [0]*20
        one_hot[aa[i]][i] = 1 
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0]*20]*(padding-len(seq_list[i]))
        feat_list.append(feat)   
    return(np.array(feat_list))

def predict_by_class(scores):
    """
    Turn prediction scores into classes.
    If score > 0.5, label the sample as 1; else 0.
    Input: scores - scores predicted by the model, 1-d array.
    Output: an array of 0s and 1s.
    """
    classes = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            classes.append(1)
        else:
            classes.append(0)
    return np.array(classes)

def base_loss(y_true, y_pred):
    """
    Strata bazowa (L_base) dla modelu, używająca binarnej krzyżowej entropii.

    Args:
        y_true (tf.Tensor): Rzeczywiste etykiety (0 lub 1).
        y_pred (tf.Tensor): Przewidywania modelu.
    Returns:
        tf.Tensor: Wartość straty bazowej.
    """

    y_true_activity = y_true[:, 0:1]
    y_true_toxicity = y_true[:, 1:2]
    
    y_pred_activity = y_pred[0]
    y_pred_toxicity = y_pred[1]

    loss_activity = tf.keras.losses.binary_crossentropy(y_true_activity, y_pred_activity)
    loss_toxicity = tf.keras.losses.binary_crossentropy(y_true_toxicity, y_pred_toxicity)
    
    return (loss_activity + loss_toxicity) / 2

def contrastive_loss(attention_activity, attention_toxicity, y_true_input_right, y_true_input_left, margin=2.0):
    # Obliczanie odległości euklidesowej
    euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(attention_activity - attention_toxicity), axis=1))
    
    # Ustalanie wartości y na podstawie etykiet
    y = tf.cast(tf.equal(y_true_input_right, y_true_input_left), dtype=tf.float32)
    print("y w constrative loss:", y)

    # Strata kontrastowa
    loss = tf.reduce_mean(
        (1 - y) * tf.square(euclidean_distance) +
        y * tf.square(tf.maximum(margin - euclidean_distance, 0))
    )
    return loss

def build_attention():
    """
    Build the model architecture for attention output
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, 
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    hidden = Attention(return_attention=True, name='Attention')(hidden)
    model = Model(inputs=inputs, outputs=hidden)
    return model

def build_amplify_architecture_attention():
    """
    Build the complete model architecture
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, 
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    attention = Attention(name='Attention', return_attention=False)(hidden)
    model = Model(inputs=inputs, outputs=attention)
    return model


def load_multi_model(model_dir_list, architecture):
    """
    Load multiple models with the same architecture in one function.
    Input: list of saved model weights files.
    Output: list of loaded models.
    """
    model_list = []
    for i in range(len(model_dir_list)):
        model = architecture()
        model.load_weights(model_dir_list[i], by_name=True)
        model_list.append(model)
    return model_list

def load_base_model():

    model_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/models/'
    models = [model_dir + "balanced" + '/AMPlify_' + "balanced" + '_model_weights_' \
              + str(i+1) + '.h5' for i in range(5)]
    for n in range(5):
        print(models[n])
    # load models for final output
    out_model = load_multi_model(models, build_amplify_architecture_attention)
    out_model[0].summary()

    return out_model[0]

class BaseModelLayer(Layer):
    """Stack of Linear layers with a sparsity regularization loss."""

    def __init__(self, y_true):
        super().__init__()
        self.model_amplify_activity = load_base_model()
        self.model_amplify_toxicity = load_base_model()
        self.y_true = y_true
        
        for layer in self.model_amplify_activity.layers:  # Freeze the layers
            layer.trainable = False
        for layer in self.model_amplify_toxicity.layers:  # Freeze the layers
            layer.trainable = False  

    def call(self, inputs):
        input_data = inputs  # Teraz przekazujemy tylko dane wejściowe

        attention_activity = self.model_amplify_activity(input_data)
        attention_toxicity = self.model_amplify_toxicity(input_data)

        # Dense layers for prediction
        concatenated = Concatenate(name='Concatenate', axis=-1)([attention_activity, attention_toxicity])

        # Wykorzystanie etykiet w funkcji straty, ale nie jako tensor wejściowy
        y_true_input_left = self.y_true[:, 0]  # Etykieta dla aktywności
        y_true_input_right = self.y_true[:, 1]  # Etykieta dla toksyczności

        #self.add_loss(contrastive_loss(attention_activity, attention_toxicity, y_true_input_left, y_true_input_right))
        return concatenated

def build_duo_model_with_custom_layer(y_true):
    """
    Build the model architecture using BaseModelLayer.
    """
    inputs_data = Input(shape=(MAX_LEN, 20), name='Input_duo')

    # Zastosowanie customowej warstwy opartej o klasę BaseModelLayer
    base_layer_output = BaseModelLayer(y_true)(inputs_data)

    # Dense layers for prediction
    prediction_activity = Dense(1, activation='sigmoid', name='Output_activity')(base_layer_output)
    prediction_toxicity = Dense(1, activation='sigmoid', name='Output_toxicity')(base_layer_output)

    # Definicja modelu
    duo_model = Model(inputs=inputs_data, outputs=[prediction_activity, prediction_toxicity])

    # Kompilacja modelu z własną funkcją straty
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    duo_model.compile(loss=base_loss, optimizer=adam, metrics=['accuracy'])

    return duo_model

def main():
    parser = argparse.ArgumentParser(description=dedent('''
        Siamense training
        ------------------------------------------------------
        Given training sets with two labels: AMP and non-AMP,
        train the AMP prediction model.    
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required fasta datasets for activity training
    parser.add_argument('-amp_act_tr', help="Training activity AMP set, fasta file", required=True)
    parser.add_argument('-non_amp_act_tr', help="Training activity non-AMP set, fasta file", required=True)

    parser.add_argument('-amp_act_te', help="Test AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-non_act_amp_te', help="Test non-AMP set, fasta file (optional)", default=None, required=False)

    # Required fasta datasets for toxicity training
    parser.add_argument('-amp_tox_tr', help="Training activity AMP set, fasta file", required=True)
    parser.add_argument('-non_amp_tox_tr', help="Training activity non-AMP set, fasta file", required=True)

    parser.add_argument('-amp_tox_te', help="Test AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-non_tox_amp_te', help="Test non-AMP set, fasta file (optional)", default=None, required=False)

    # Required output directory and model name
    parser.add_argument('-out_dir', help="Output directory", required=True)
    parser.add_argument('-model_name', help="File name of trained model weights", required=True)
    
    args = parser.parse_args()
    
    #load training sets: activity and toxicity
    amp_act_train = []
    non_amp_act_train = []
    for seq_record in SeqIO.parse(args.amp_act_tr, 'fasta'):
        # "../data/AMPlify_AMP_train_common.fa"
        amp_act_train.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_act_tr, 'fasta'):
        # "../data/AMPlify_non_AMP_train_balanced.fa"
        non_amp_act_train.append(str(seq_record.seq))

    # sequences for training sets (activity)
    train_act_seq = amp_act_train + non_amp_act_train

    train_act_seq = train_act_seq[:100]
    # set labels for training sequences
    y_act_train = np.array([1]*len(amp_act_train) + [0]*len(non_amp_act_train))

    y_act_train=y_act_train[:100]

    # shuffle training set
    train_act = list(zip(train_act_seq, y_act_train))
    random.Random(123).shuffle(train_act)
    train_act_seq, y_act_train = zip(*train_act) 
    train_act_seq = list(train_act_seq)
    y_act_train = np.array((y_act_train))

    AMP_tox_train = []
    non_AMP_tox_train = []
    for seq_record in SeqIO.parse(args.amp_tox_tr, 'fasta'):
        # "../data_toxicity_AMPDeep/train_toxicity_AMP.fa"
        AMP_tox_train.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_tox_tr, 'fasta'):
        # "../data_toxicity_AMPDeep/train_toxicity_nonAMP.fa"
        non_AMP_tox_train.append(str(seq_record.seq))

    # sequences for training sets (toxicity)
    train_tox_seq = AMP_tox_train + non_AMP_tox_train

    train_tox_seq = train_tox_seq[:100]
    # set labels for training sequences (toxicity)
    y_tox_train = np.array([0]*len(train_tox_seq))

    y_tox_train = y_tox_train[:100]

    i=0
    for seq in train_act_seq:
        if seq in AMP_tox_train:
            y_tox_train[i] = 1
        i+=1

    y_combined = np.array(list(zip(y_act_train, y_tox_train)))

    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_union = one_hot_padding(train_act_seq, MAX_LEN)
    
    indv_pred_train = [] # list of predicted scores for individual models on the training set

    ### Train siamese model for activity and toxicity
    ensemble_number = 2 # number of training subsets for ensemble
    ensemble = StratifiedKFold(n_splits=ensemble_number, shuffle=True, random_state=50)
    save_file_num = 0
    

    for tr_ens, te_ens in ensemble.split(X_union, y_act_train):
        # Trenuj model z danymi i etykietami
        model = build_duo_model_with_custom_layer(y_combined[tr_ens])

        early_stopping = EarlyStopping(monitor='val_accuracy',  min_delta=0.001, patience=50, restore_best_weights=True)
        model.fit(X_union[tr_ens], np.array(y_combined[tr_ens]), epochs=1, batch_size=20, 
                validation_data=(X_union[te_ens], y_combined[te_ens]), 
                verbose=2, initial_epoch=0, callbacks=[early_stopping])

        # Predykcja (teraz tylko dane są potrzebne)
        temp_pred_train = model.predict(X_union)
        temp_pred_train_activity = temp_pred_train[0].flatten()
        temp_pred_train_toxicity = temp_pred_train[1].flatten()

        indv_pred_train.append((temp_pred_train_activity, temp_pred_train_toxicity))

        train_pred_classes_activity = predict_by_class(temp_pred_train_activity)
        train_pred_classes_toxicity = predict_by_class(temp_pred_train_toxicity)

        # Calculate training accuracy for activity and toxicity
        train_acc_activity = accuracy_score(y_act_train, train_pred_classes_activity)
        train_acc_toxicity = accuracy_score(y_tox_train, train_pred_classes_toxicity)
        train_acc_combined = (train_acc_activity + train_acc_toxicity) / 2

        # Predicting on the validation set
        temp_pred_val = model.predict(X_union[te_ens])
        temp_pred_val_activity = temp_pred_val[0].flatten()
        temp_pred_val_toxicity = temp_pred_val[1].flatten()

        # Convert scores to classes using the provided function
        val_pred_classes_activity = predict_by_class(temp_pred_val_activity)
        val_pred_classes_toxicity = predict_by_class(temp_pred_val_toxicity)

        # Calculate validation accuracy for activity and toxicity
        val_acc_activity = accuracy_score(y_act_train[te_ens], val_pred_classes_activity)
        val_acc_toxicity = accuracy_score(y_tox_train[te_ens], val_pred_classes_toxicity)
        val_acc_combined = (val_acc_activity + val_acc_toxicity) / 2

        '''
        save_file_num = save_file_num + 1
        save_dir = args.out_dir + '/' + args.model_name + '_' + str(save_file_num) + '.h5'
        save_dir_wt = args.out_dir + '/' + args.model_name + '_weights_' + str(save_file_num) + '.h5'
        model.save(save_dir) #save
        model.save_weights(save_dir_wt) #save
        '''
        
        print('*************************** current model ***************************')
        print('current train acc (activity): ', train_acc_activity)
        print('current train acc (toxicity): ', train_acc_toxicity)
        print('current train acc (combined): ', train_acc_combined)
        print('current val acc (activity): ', val_acc_activity)
        print('current val acc (toxicity): ', val_acc_toxicity)
        print('current val acc (combined): ', val_acc_combined) 
        

if __name__ == "__main__":
    main()
    