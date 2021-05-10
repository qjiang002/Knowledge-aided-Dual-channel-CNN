import numpy as np
import re
import os


def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_lexicon(positive_lexicon_file, negative_lexicon_file, negation_lexicon_file, intensifier_lexicon_file):
    pos_list = list(open(positive_lexicon_file, "r", encoding='utf-8').readlines())
    pos_list = [clean_str(s.strip()) for s in pos_list]
    neg_list = list(open(negative_lexicon_file, "r", encoding='utf-8').readlines())
    neg_list = [clean_str(s.strip()) for s in neg_list]
    negation_list = list(open(negation_lexicon_file, "r", encoding='utf-8').readlines())
    negation_list = [clean_str(s.strip()) for s in negation_list]
    intensifier_list = list(open(intensifier_lexicon_file, "r", encoding='utf-8').readlines())
    intensifier_list = [clean_str(s.strip()) for s in intensifier_list]

    return pos_list, neg_list, negation_list, intensifier_list

def load_MR(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_ARSC(file_dir, train=True):
    
    train_examples = list(open(os.path.join(file_dir, 'train.txt'), 'r', encoding='utf-8').readlines())
    train_text = []
    train_label = []
    pos_num = 0
    neg_num = 0
    for s in train_examples:
        s = s.strip().split('\t')
        if len(s)==2 and len(s[1])>0 and len(s[0])>0:
            l = int(s[1])
            if l == -1 and neg_num<5000:
                neg_num += 1
                train_text.append(clean_str(s[0]))
                train_label.append([1, 0])
            elif l == 1 and pos_num<5000:
                pos_num += 1
                train_text.append(clean_str(s[0]))
                train_label.append([0, 1])

    assert len(train_text) == len(train_label)
    return [train_text, train_label]

def load_yelp(file_dir, train=True):

    train_examples = list(open(os.path.join(file_dir, 'train_mini.txt'), 'r', encoding='utf-8').readlines())
    train_text = []
    train_label = []
    pos_num = 0
    neg_num = 0
    for s in train_examples:
        s = s.strip().split('\t')
        if len(s)==2 and len(s[1])>0 and len(s[0])>0:
            
            l = int(s[1])
            if l == 1 and pos_num<5000:
                pos_num += 1
                train_text.append(clean_str(s[0]))
                train_label.append([1, 0])
            elif l == -1 and neg_num<5000:
                neg_num += 1
                train_text.append(clean_str(s[0]))
                train_label.append([0, 1])

    assert len(train_text) == len(train_label)
    return [train_text, train_label]

def load_twitter(file_dir, train=True):

    train_examples = list(open(os.path.join(file_dir, 'train.txt'), 'r', encoding='utf-8').readlines())
    train_text = []
    train_label = []
    pos_num = 0
    neg_num = 0
    for s in train_examples:
        s = s.strip().split('\t')
        if len(s)==2 and len(s[1])>0 and len(s[0])>0:
            
            l = int(s[1])
            if l == 1 and pos_num<5000:
                pos_num += 1
                train_text.append(clean_str(s[0]))
                train_label.append([1, 0])
            elif l == -1 and neg_num<5000:
                neg_num += 1
                train_text.append(clean_str(s[0]))
                train_label.append([0, 1])
    
    assert len(train_text) == len(train_label)
    return [train_text, train_label]

def load_COVID_Chinese(file_dir, train=True):
    if train:
        train_examples = list(open(os.path.join(file_dir, 'train.txt'), 'r', encoding='utf-8').readlines())
        train_text = []
        train_label = []
        for s in train_examples:
            s = s.strip().split('\t')
            if len(s)==2 and len(s[0])>0 and s[0]!='nan' and (s[1]=='-1' or s[1]=='0' or s[1]=='1'):
                train_text.append(s[0])
                l = int(s[1])
                if l == -1:
                    train_label.append([1, 0, 0])
                elif l == 0:
                    train_label.append([0, 1, 0])
                elif l == 1:
                    train_label.append([0, 0, 1])
        

        dev_examples = list(open(os.path.join(file_dir, 'dev.txt'), 'r', encoding='utf-8').readlines())
        dev_text = []
        dev_label = []
        for s in dev_examples:
            s = s.strip().split('\t')
            if len(s)==2 and len(s[0])>0 and s[0]!='nan' and (s[1]=='-1' or s[1]=='0' or s[1]=='1'):
                dev_text.append(s[0])
                l = int(s[1])
                if l == -1:
                    dev_label.append([1, 0, 0])
                elif l == 0:
                    dev_label.append([0, 1, 0])
                elif l == 1:
                    dev_label.append([0, 0, 1])

        assert len(train_text) == len(train_label)
        assert len(dev_text) == len(dev_label)
        return [train_text, train_label, dev_text, dev_label]
    else:
        test_examples = list(open(os.path.join(file_dir, 'test.txt'), 'r', encoding='utf-8').readlines())
        test_text = []
        test_label = []
        for s in test_examples:
            s = s.strip().split('\t')
            if len(s)==2 and len(s[0])>0 and s[0]!='nan' and (s[1]=='-1' or s[1]=='0' or s[1]=='1'):
                test_text.append(s[0])
                l = int(s[1])
                if l == -1:
                    test_label.append([1, 0, 0])
                elif l == 0:
                    test_label.append([0, 1, 0])
                elif l == 1:
                    test_label.append([0, 0, 1])
        assert len(test_text) == len(test_label)
        return [test_text, test_label]

def load_SST5(file_dir):

    train_examples = list(open(os.path.join(file_dir, 'sst_train.txt'), 'r', encoding='utf-8').readlines())
    train_text = []
    train_label = []
    for s in train_examples:
        s = s.strip().split('\t')
        if len(s)==2 and len(s[1])>0 and len(s[0])>0:
            train_text.append(clean_str(s[1]))
            l = int(s[0].replace('__label__',''))
            if l == 1:
                train_label.append([1, 0, 0, 0, 0])
            elif l == 2:
                train_label.append([0, 1, 0, 0, 0])
            elif l == 3:
                train_label.append([0, 0, 1, 0, 0])
            elif l == 4:
                train_label.append([0, 0, 0, 1, 0])
            elif l == 5:
                train_label.append([0, 0, 0, 0, 1])
    

    dev_examples = list(open(os.path.join(file_dir, 'sst_dev.txt'), 'r', encoding='utf-8').readlines())
    dev_text = []
    dev_label = []
    for s in dev_examples:
        s = s.strip().split('\t')
        if len(s)==2 and len(s[1])>0 and len(s[0])>0:
            dev_text.append(clean_str(s[1]))
            l = int(s[0].replace('__label__',''))
            if l == 1:
                dev_label.append([1, 0, 0, 0, 0])
            elif l == 2:
                dev_label.append([0, 1, 0, 0, 0])
            elif l == 3:
                dev_label.append([0, 0, 1, 0, 0])
            elif l == 4:
                dev_label.append([0, 0, 0, 1, 0])
            elif l == 5:
                dev_label.append([0, 0, 0, 0, 1])

    assert len(train_text) == len(train_label)
    assert len(dev_text) == len(dev_label)

    test_examples = list(open(os.path.join(file_dir, 'sst_test.txt'), 'r', encoding='utf-8').readlines())
    test_text = []
    test_label = []
    for s in test_examples:
        s = s.strip().split('\t')
        if len(s)==2 and len(s[1])>0 and len(s[0])>0:
            test_text.append(clean_str(s[1]))
            l = int(s[0].replace('__label__',''))
            if l == 1:
                test_label.append([1, 0, 0, 0, 0])
            elif l == 2:
                test_label.append([0, 1, 0, 0, 0])
            elif l == 3:
                test_label.append([0, 0, 1, 0, 0])
            elif l == 4:
                test_label.append([0, 0, 0, 1, 0])
            elif l == 5:
                test_label.append([0, 0, 0, 0, 1])

    assert len(test_text) == len(test_label)
    return [train_text, train_label, dev_text, dev_label, test_text, test_label]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
