#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script is for  siamense model training, and testing the performance if a test set is specified

@author: Natalia Kowalczyk
"""
from keras.layers import Input, Masking, Bidirectional, LSTM, Dropout, Concatenate, Dense, GlobalMaxPooling1D
from layers import*
from AMPlify import*
from train_amplify import*
from keras.layers import Concatenate
from keras.backend import int_shape

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

def build_siamese_model():
    """
    Build the model architecture for base siamese model with attention, returning probabilities for activity and toxicity.
    """
    inputs_right = Input(shape=(MAX_LEN, 20), name='Input_right')
    print(int_shape(inputs_right))
    inputs_left = Input(shape=(MAX_LEN, 20), name='Input_left')
    print(int_shape(inputs_left))

    # Right branch
    masking_right = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking_right')(inputs_right)
    print(int_shape(masking_right))
    hidden_right1 = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM_right')(masking_right)
    print(int_shape(hidden_right1))
    hidden_right2 = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, return_multi_attention=False, name='Multi-Head-Attention_right')(hidden_right1)
    print(int_shape(hidden_right2))
    hidden_right3 = Dropout(0.2, name='Dropout_1_right')(hidden_right2)
    print(int_shape(hidden_right3))
    attention_right = Attention(return_attention=True, name='Attention_right')(hidden_right3)
    print(int_shape(attention_right))

    # Left branch
    masking_left = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking_left')(inputs_left)
    hidden_left1 = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM_left')(masking_left)
    hidden_left2 = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, return_multi_attention=False, name='Multi-Head-Attention_left')(hidden_left1)
    hidden_left3 = Dropout(0.2, name='Dropout_1_left')(hidden_left2)
    attention_left = Attention(return_attention=True, name='Attention_left')(hidden_left3)

    # Concatenate the two attention layers
    concatenated = Concatenate(name='Concatenate')([attention_right, attention_left])
    print(int_shape(concatenated))

    # Apply Global Max Pooling to get a single output per sequence
    pooled_output = GlobalMaxPooling1D()(concatenated)
    print(int_shape(pooled_output))

    # Dense layers for prediction
    prediction_activity = Dense(1, activation='sigmoid', name='Output_activity')(pooled_output)
    prediction_toxicity = Dense(1, activation='sigmoid', name='Output_toxicity')(pooled_output)
    
    print("prediction_activity", int_shape(prediction_activity)) 
    print("prediction_toxicity", int_shape(prediction_toxicity))
    # Define the model with two outputs
    siamese_model = Model(inputs=[inputs_right, inputs_left], outputs=[prediction_activity, prediction_toxicity])

    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    siamese_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return siamese_model

def avg_binary_crossentropy(y_true, y_pred):
    """
    Calculate the average binary crossentropy.
    """
    return K.mean(K.binary_crossentropy(y_true, y_pred))

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

    ensemble_number = 2 # number of training subsets for ensemble
    ensemble = StratifiedKFold(n_splits=ensemble_number, shuffle=True, random_state=50)
    save_file_num = 0
    

    for tr_ens, te_ens in ensemble.split(X_act_train, y_act_train):
        model = build_siamese_model()

        early_stopping = EarlyStopping(monitor='val_accuracy',  min_delta=0.001, patience=50, restore_best_weights=True)
        #model.fit([X_act_train[tr_ens],X_tox_train[tr_ens]] , [np.array(y_act_train[tr_ens]), np.array(y_tox_train[tr_ens])], epochs=1000, batch_size=32, 
                      #validation_data=[(X_act_train[te_ens], y_act_train[te_ens]), (X_tox_train[te_ens], y_tox_train[te_ens])], verbose=2, initial_epoch=0, callbacks=[early_stopping])
            # Trenuj model tylko na danych aktywności (X_act_train)
        model.fit(
            [X_act_train[tr_ens], X_tox_train[tr_ens]],  # Dwa razy te same dane, zgodnie z architekturą
            y_act_train[tr_ens],  # Odpowiednie etykiety
            epochs=1000,batch_size=32,validation_data=([X_act_train[te_ens], X_act_train[te_ens]], y_act_train[te_ens]),verbose=2,initial_epoch=0,callbacks=[early_stopping])
        print("funkcja model fit przeszla")



        temp_pred_train_activity, temp_pred_train_toxicity = model.predict([X_act_train, X_tox_train])

        # Jeśli chcesz spłaszczyć oba wyniki:
        temp_pred_train_activity = temp_pred_train_activity.flatten()
        temp_pred_train_toxicity = temp_pred_train_toxicity.flatten()

        # Zapisz wyniki predykcji dla aktualnego modelu w ensemble
        indv_pred_train.append((temp_pred_train_activity, temp_pred_train_toxicity))
        

        save_file_num = save_file_num + 1
        save_dir = args.out_dir + '/' + args.model_name + '_' + str(save_file_num) + '.h5'
        save_dir_wt = args.out_dir + '/' + args.model_name + '_weights_' + str(save_file_num) + '.h5'
        model.save(save_dir) #save
        model.save_weights(save_dir_wt) #save


        #Obliczenie predykcji i dokładności dla danych treningowych i walidacyjnych
        temp_pred_train_activity, temp_pred_train_toxicity = model.predict([X_act_train[tr_ens], X_tox_train[tr_ens]])
        temp_pred_val_activity, temp_pred_val_toxicity = model.predict([X_act_train[te_ens], X_tox_train[te_ens]])

        temp_pred_class_train_activity = predict_by_class(temp_pred_train_activity.flatten())
        temp_pred_class_train_toxicity = predict_by_class(temp_pred_train_toxicity.flatten())

        temp_pred_class_val_activity = predict_by_class(temp_pred_val_activity.flatten())
        temp_pred_class_val_toxicity = predict_by_class(temp_pred_val_toxicity.flatten())

        # Oblicz dokładność
        train_acc_activity = accuracy_score(y_act_train[tr_ens], temp_pred_class_train_activity)
        val_acc_activity = accuracy_score(y_act_train[te_ens], temp_pred_class_val_activity)

        train_acc_toxicity = accuracy_score(y_tox_train[tr_ens], temp_pred_class_train_toxicity)
        val_acc_toxicity = accuracy_score(y_tox_train[te_ens], temp_pred_class_val_toxicity)

        train_acc_combined = (train_acc_activity + train_acc_toxicity) / 2
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
    