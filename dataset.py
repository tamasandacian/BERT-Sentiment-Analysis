
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd

class Dataset(object):
    """ Class for pre-processing dataset as Pandas DataFrame 
    
    :param method: classification method
    :param tokenizer: BERT pre-trained tokenizer
    :param label_col: label column name in DataFrame
    :param text_col: text column name in DataFrame
    :param test_size: test DataFrame size
    :param random_state: set for reproducibility
    :param batch_size: training chunk samples;  recommended 16 or 32
    """
    def __init__(self, method="BERT", tokenizer=None, label_col="label", text_col="text", 
                 test_size=0.20, random_state=42, batch_size=32):
        
        self.method = method
        self.tokenizer = tokenizer
        self.label_col = label_col
        self.text_col = text_col
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size

    def preprocessing(self, df, max_length):
        """ Pre-process dataframe and generate train, test data 
        
        :param df: DataFrame
        :param max_length: max sequence length required by BERT pre-trained model
        """
        if self.method == "BERT":
            
            # generate label_index, index_label, class_names
            label_index = self.get_label_index(df)
            index_label = self.get_index_label(label_index)
            class_names = self.class_names(index_label)

            # generate numerical column feature from original label column
            df['label_no'] = df[self.label_col].replace(label_index)

            # separate input features and target
            y = df['label_no']
            X = df.drop('label_no', axis=1)
            
            # split dataframe into train, test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

            # concatenate training data back together
            X = pd.concat([X_train, y_train], axis=1)
            # apply sampling to fix imbalanced classes in train data
            X = self.sampling(X)     
            X = shuffle(X)
            
            # concatenate test data back together
            Y = pd.concat([X_test, y_test])
        
            # tokenize & encode sequences for train, validation sets
            tokens_train = self.tokenizer.batch_encode_plus(
                X['text'].tolist(), max_length=max_length, 
                padding=True, return_attention_mask=True, 
                truncation=True
            )
            
            tokens_test = self.tokenizer.batch_encode_plus(
                X_test['text'].tolist(), max_length=max_length, 
                padding=True, return_attention_mask=True, 
                truncation=True
            )

            # convert lists to tensors
            train_seq = torch.tensor(tokens_train['input_ids'])
            train_mask = torch.tensor(tokens_train['attention_mask'])
            train_y = torch.tensor(X['label_no'].tolist())
           
            test_seq = torch.tensor(tokens_test['input_ids'])
            test_mask = torch.tensor(tokens_test['attention_mask'])
            test_y = torch.tensor(y_test.tolist())

            # convert train, test tensors to dataloader
            train_data = TensorDataset(train_seq, train_mask, train_y)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

            test_data = TensorDataset(test_seq, test_mask, test_y)
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)

            self.X = X                          # train dataset
            self.Y = Y                          # test dataset
            self.label_index = label_index      # label-index mapping
            self.index_label = index_label      # index-label mapping
            self.class_names = class_names
            self.train_text = X['text'].tolist()
            self.test_text = X_test['text'].tolist()
            self.train_seq = train_seq
            self.test_seq = test_seq
            self.train_mask = train_mask
            self.test_mask = test_mask
            self.train_y = train_y
            self.test_y = test_y
            self.train_dataloader = train_dataloader
            self.test_dataloader = test_dataloader

        else:
            raise Exception('method not exist')

    def get_label_index(self, df):
        """ Generate label-index mapping from dataframe using label colname 
        
        :param df: DataFrame
        :return: dictionary
        """
        labels = df[self.label_col].unique()
        label_index = dict()
        for index, label in enumerate(labels):
            label_index[label] = index
        return label_index
    
    def get_index_label(self, label_index):
        """ Generate index-label mapping reversing key, values in label_index dictionary
        
        :param label_index: label-index dictionary 
        :return: dictionary
        """
        index_label = {v: k for k, v in label_index.items()}
        return index_label

    def class_names(self, index_label):
        """ Get list of class names from index_label dictionary 
        
        :param index_label: index-label dictionary
        :return: list
        """
        class_names = list(index_label.values())
        return class_names
    
    def get_label_count(self, df):
        """ Get count for each label in a dictionary 
        
        :param df: DataFrame
        :return: dictionary
        Sample:
                {
                    'negative': 2971,
                    'positive': 2966,
                    'neutral': 2144
                }
        """
        label_count = df[self.label_col].value_counts().to_dict()
        return label_count

    def get_avg_label_count(self, df):
        """ Get the average count across all labels in dataframe 
        
        :param df: DataFrame
        :return: int
        """
        avg_count = int( df[self.label_col].value_counts().mean())
        return avg_count

    def sampling(self, df):
        """ Sampling class-imbalance using the average records across all labels
        
        :param df: DataFrame
        :return: DataFrame
        """
        label_count = self.get_label_count(df)
        avg_count = self.get_avg_label_count(df)

        frames = []
        for label, count in label_count.items():
            if count >= avg_count:
                # downsample dataframe to the mean across all labels
                df_label = df[df[self.label_col] == label]
                df_label = df_label.head(avg_count)
                frames.append(df_label)
            else:
                # oversample dataframe to the mean across all labels
                df_label = df[df[self.label_col] == label]
                df_label = df_label.sample(avg_count, replace=True)
                frames.append(df_label)

        df = pd.concat(frames)
        return df

