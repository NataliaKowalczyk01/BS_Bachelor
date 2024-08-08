#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script is for  siamense model training, and testing the performance if a test set is specified

@author: Natalia Kowalczyk
"""
from layers import*
from AMPlify import*
from train_amplify import*
from keras.layers import Concatenate

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

def build_model():
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
    prediction = Dense(1, activation='sigmoid', name='Output')(hidden)
    model = Model(inputs=inputs, outputs=prediction)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #best
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def build_siamese_model():
    """
    Build the model architecture for base siamese model with attention.
    """
    inputs_right = Input(shape=(MAX_LEN, 20), name='Input_right')
    inputs_left = Input(shape=(MAX_LEN, 20), name='Input_left')

    # Right branch
    masking_right = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking_right')(inputs_right)
    hidden_right1 = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM_right')(masking_right)
    hidden_right2 = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, return_multi_attention=False, name='Multi-Head-Attention_right')(hidden_right1)
    hidden_right3 = Dropout(0.2, name='Dropout_1_right')(hidden_right2)
    attention_right, _ = Attention(return_attention=True, name='Attention_right')(hidden_right3)

    # Left branch
    masking_left = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking_left')(inputs_left)
    hidden_left1 = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM_left')(masking_left)
    hidden_left2 = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, return_multi_attention=False, name='Multi-Head-Attention_left')(hidden_left1)
    hidden_left3 = Dropout(0.2, name='Dropout_1_left')(hidden_left2)
    attention_left, _ = Attention(return_attention=True, name='Attention_left')(hidden_left3)

    # Concatenate the two attention layers
    concatenated = Concatenate(name='Concatenate')([attention_right, attention_left])

    # Dense layer for prediction
    prediction = Dense(1, activation='sigmoid', name='Output')(concatenated)
    siamese_model = Model(inputs=[inputs_right, inputs_left], outputs=prediction)

    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    siamese_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return siamese_model


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
    AMP_act_train = []
    non_AMP_act_train = []
    for seq_record in SeqIO.parse(args.amp_act_tr, 'fasta'):
        # "../data/AMPlify_AMP_train_common.fa"
        AMP_act_train.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_act_tr, 'fasta'):
        # "../data/AMPlify_non_AMP_train_balanced.fa"
        non_AMP_act_train.append(str(seq_record.seq))

    # sequences for training sets (activity)
    train_act_seq = AMP_act_train + non_AMP_act_train    
    # set labels for training sequences
    y_act_train = np.array([1]*len(AMP_act_train) + [0]*len(non_AMP_act_train))

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
    # set labels for training sequences (toxicity)
    y_tox_train = np.array([1]*len(AMP_tox_train) + [0]*len(non_AMP_tox_train))

    # shuffle training set
    train_tox = list(zip(train_tox_seq, y_tox_train))
    random.Random(123).shuffle(train_tox)
    train_tox_seq, y_tox_train = zip(*train_tox) 
    train_tox_seq = list(train_tox_seq)
    y_tox_train = np.array((y_tox_train))
    


    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_act_train = one_hot_padding(train_act_seq, MAX_LEN)
    X_tox_train = one_hot_padding(train_tox_seq, MAX_LEN)
    
    indv_pred_train = [] # list of predicted scores for individual models on the training set
         
    ### Train siamese model for activity and toxicity

    ensemble_number = 5 # number of training subsets for ensemble
    ensemble = StratifiedKFold(n_splits=ensemble_number, shuffle=True, random_state=50)
    save_file_num = 0
    
    for tr_ens, te_ens in ensemble.split(X_act_train, y_act_train):
        model = build_siamese_model()

        early_stopping = EarlyStopping(monitor='val_accuracy',  min_delta=0.001, patience=50, restore_best_weights=True)
        model.fit(X_act_train[tr_ens], np.array(y_act_train[tr_ens]), epochs=1000, batch_size=32, 
                      validation_data=(X_act_train[te_ens], y_act_train[te_ens]), verbose=2, initial_epoch=0, callbacks=[early_stopping])

        temp_pred_train = model.predict(X_act_train).flatten() # predicted scores on the [whole] training set from the current model
        indv_pred_train.append(temp_pred_train)


        save_file_num = save_file_num + 1
        save_dir = args.out_dir + '/' + args.model_name + '_' + str(save_file_num) + '.h5'
        save_dir_wt = args.out_dir + '/' + args.model_name + '_weights_' + str(save_file_num) + '.h5'
        model.save(save_dir) #save
        model.save_weights(save_dir_wt) #save

        # training and validation accuracy for the current model
        temp_pred_class_train_curr = predict_by_class(model.predict(X_act_train[tr_ens]).flatten())
        temp_pred_class_val = predict_by_class(model.predict(X_act_train[te_ens]).flatten())

        print('*************************** current model ***************************')
        print('current train acc: ', accuracy_score(y_act_train[tr_ens], temp_pred_class_train_curr))
        print('current val acc: ', accuracy_score(y_act_train[te_ens], temp_pred_class_val)) 
        

if __name__ == "__main__":
    main()
    