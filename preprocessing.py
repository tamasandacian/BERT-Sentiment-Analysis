import torch

class Preprocessing(object):
    """ Class for pre-processing textual content
    
    :param tokenizer: BERT pre-trained tokenizer
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def tokenize_and_encode(self, texts, max_length):
        """ Tokenize and encode text 
        
        :param texts: list of documents
        :param max_length: required max_length in text by BERT pre-trained model
        :return: input_ids, attention_mask
        """
        tokens_texts = self.tokenizer.batch_encode_plus(
            texts, 
            max_length=max_length, 
            padding=True, 
            return_attention_mask=True, 
            truncation=True
        )
        input_ids = tokens_texts["input_ids"]
        attention_mask = tokens_texts["attention_mask"]
        return input_ids, attention_mask
    
    def text_preprocessing(self, texts, max_length=128):
        """ Generate input_ids, attention_mask from a given text 
        
        :param texts: list of documents
        :param max_length: required max_length in text by BERT pre-trained model
        :return: torch input_ids, attention_mask 
        """
        input_ids, attention_mask = self.tokenize_and_encode(texts, max_length)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, attention_mask