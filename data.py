import pickle
from typing import Dict

import numpy as np


def load_mosi():
    with open('./dataset/mosi_data.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        return data


def extract_vision_data(data: Dict):
    return data['train']['vision'], data['valid']['vision'], data['test']['vision']


def extract_audio_data(data: Dict):
    return data['train']['audio'], data['valid']['audio'], data['test']['audio']


def extract_text_data(data: Dict):
    return data['train']['text'], data['valid']['text'], data['test']['text']


def extract_labels_data(data: Dict):
    return data['train']['labels'], data['valid']['labels'], data['test']['labels']


def create_mask(train_data, test_data, train_length, test_length):
    '''
    # Arguments
        train, test data (any one modality (text, audio or video)), utterance lengths in train, test videos

    # Returns
        mask for train and test data
    '''

    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    return train_mask, test_mask


