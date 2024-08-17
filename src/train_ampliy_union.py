#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2: amplify_union
This script is for model training, and testing the performance if a test set is specified
"""

import argparse
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
from layers import Attention, MultiHeadAttention
from keras.models import Model
from keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers.legacy import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf

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

def combined_binary_crossentropy(y_true, y_pred):
    y_true_activity = y_true[:, 0:1]
    y_true_toxicity = y_true[:, 1:2]
    
    y_pred_activity = y_pred[0]
    y_pred_toxicity = y_pred[1]
    
    loss_activity = tf.keras.losses.binary_crossentropy(y_true_activity, y_pred_activity)
    loss_toxicity = tf.keras.losses.binary_crossentropy(y_true_toxicity, y_pred_toxicity)
    
    return (loss_activity + loss_toxicity) / 2

def build_union_model():
    """
    Build and compile the model.
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, 
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    hidden = Attention(name='Attention')(hidden)
    # Two outputs for activity and toxicity
    prediction_activity = Dense(1, activation='sigmoid', name='Output_1')(hidden)
    prediction_toxicity = Dense(1, activation='sigmoid', name='Output_2')(hidden)
    model = Model(inputs=inputs, outputs=[prediction_activity, prediction_toxicity])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #best
    model.compile(loss=combined_binary_crossentropy, optimizer=adam, metrics=['accuracy'])
    return model

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
    # set labels for training sequences
    y_act_train = np.array([1]*len(amp_act_train) + [0]*len(non_amp_act_train))

    # shuffle training set
    train_act = list(zip(train_act_seq, y_act_train))
    random.Random(123).shuffle(train_act)
    train_act_seq, y_act_train = zip(*train_act) 
    train_act_seq = list(train_act_seq)
    y_act_train = np.array((y_act_train))
    print(train_act_seq, y_act_train)

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
    # set labels for training sequences (toxicity)
    y_tox_train = np.array([0]*len(train_tox_seq))

    i=0
    for seq in train_act_seq:
        if seq in AMP_tox_train:
            y_tox_train[i] = 1
        i+=1

    y_combined = np.array(list(zip(y_act_train, y_tox_train)))

    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_act_train = one_hot_padding(train_act_seq, MAX_LEN)
    
    indv_pred_train = [] # list of predicted scores for individual models on the training set

    ### Train siamese model for activity and toxicity
    ensemble_number = 2 # number of training subsets for ensemble
    ensemble = StratifiedKFold(n_splits=ensemble_number, shuffle=True, random_state=50)
    save_file_num = 0
    

    for tr_ens, te_ens in ensemble.split(X_act_train, y_act_train):
        model = build_union_model()
        print("Model został zbudowany")

        early_stopping = EarlyStopping(monitor='val_accuracy',  min_delta=0.001, patience=50, restore_best_weights=True)
        model.fit(X_act_train[tr_ens], np.array(y_combined[tr_ens]), epochs=10, batch_size=32, 
                      validation_data=(X_act_train[te_ens], y_combined[te_ens]), verbose=2, initial_epoch=0, callbacks=[early_stopping])

        # Predicting on the training set
        temp_pred_train = model.predict(X_act_train)
        temp_pred_train_activity = temp_pred_train[0].flatten()
        temp_pred_train_toxicity = temp_pred_train[1].flatten()

        indv_pred_train.append((temp_pred_train_activity, temp_pred_train_toxicity))


        save_file_num = save_file_num + 1
        save_dir = args.out_dir + '/' + args.model_name + '_' + str(save_file_num) + '.h5'
        save_dir_wt = args.out_dir + '/' + args.model_name + '_weights_' + str(save_file_num) + '.h5'
        model.save(save_dir) #save
        model.save_weights(save_dir_wt) #save

        train_pred_classes_activity = predict_by_class(temp_pred_train_activity)
        train_pred_classes_toxicity = predict_by_class(temp_pred_train_toxicity)

        # Calculate training accuracy for activity and toxicity
        train_acc_activity = accuracy_score(y_act_train, train_pred_classes_activity)
        train_acc_toxicity = accuracy_score(y_tox_train, train_pred_classes_toxicity)
        train_acc_combined = (train_acc_activity + train_acc_toxicity) / 2

        # Predicting on the validation set
        temp_pred_val = model.predict(X_act_train[te_ens])
        temp_pred_val_activity = temp_pred_val[0].flatten()
        temp_pred_val_toxicity = temp_pred_val[1].flatten()

        # Convert scores to classes using the provided function
        val_pred_classes_activity = predict_by_class(temp_pred_val_activity)
        val_pred_classes_toxicity = predict_by_class(temp_pred_val_toxicity)

        # Calculate validation accuracy for activity and toxicity
        val_acc_activity = accuracy_score(y_act_train[te_ens], val_pred_classes_activity)
        val_acc_toxicity = accuracy_score(y_tox_train[te_ens], val_pred_classes_toxicity)
        val_acc_combined = (val_acc_activity + val_acc_toxicity) / 2

        
        print('*************************** current model ***************************')
        print('current train acc (activity): ', train_acc_activity)
        print('current train acc (toxicity): ', train_acc_toxicity)
        print('current train acc (combined): ', train_acc_combined)
        print('current val acc (activity): ', val_acc_activity)
        print('current val acc (toxicity): ', val_acc_toxicity)
        print('current val acc (combined): ', val_acc_combined) 
        

if __name__ == "__main__":
    main()
    