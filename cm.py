from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import Dataset
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CM(object):
    """ Class for generating classification model 
    """
    def __init__(self, method="BERT", epochs=4, batch_size=32, random_state=42, lr=2e-5, eps=1e-8, 
                text_col="text", label_col="label", test_size=0.15, pre_trained_name='bert-base-uncased'):

        self.method = method
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.lr = lr
        self.eps = eps
        self.text_col = text_col
        self.label_col = label_col
        self.test_size = test_size
        self.pre_trained_name = pre_trained_name
        self.device = device

    def fit(self, df, max_length):
        """ Generate classification model from a given dataframe 
        
        :param df: dataframe
        :param max_length: max sequence length required by BERT pre-trained model
        """
        if self.method == "BERT":
        
            # define tokenizer, bert model from pretrained
            tokenizer = BertTokenizerFast.from_pretrained(
                self.pre_trained_name, 
                do_lower=True
            )
            
            # define dataset and process df with max_length sequence
            dataset = Dataset(
                method=self.method, tokenizer=tokenizer, label_col=self.label_col, text_col=self.text_col, 
                test_size=self.test_size, random_state=self.random_state, batch_size=self.batch_size
            )

            dataset.preprocessing(df, max_length)

            # create BERT model
            model = BertForSequenceClassification.from_pretrained(
                self.pre_trained_name,
                num_labels=len(dataset.label_index),
                output_attentions=False,
                output_hidden_states=False
            )
            model.to(device)
            
            # define optimizer, scheduler 
            optimizer = AdamW(model.parameters(), lr=self.lr, eps=self.eps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=len(dataset.train_dataloader) * self.epochs
            )

            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.dataset = dataset
        else:
            raise Exception('method not exists')