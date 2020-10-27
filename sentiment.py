import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizerFast
from preprocessing import Preprocessing
from shared.utils import load_from_json
from functools import lru_cache
from utility import Utility
import logging
import os

abs_path = os.path.abspath(os.path.dirname(__file__)) + "/output"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Sentiment(object):
    """ Class for predicting sentiment label in new documents

    :param lang_code: language model
    :param method: classification method
    :param version: model version number
    :param clean_text: boolean flag for cleaning text
    :param min_words: min number of words for prediction
    :param min_conf_score: minimum confidence threshold
    :param max_length: max sequence length required by BERT pre-trained model
    :param top_k: top k predictions
    :param pre_trained_name: pre-trained model name
    """
    def __init__(self, lang_code, method="BERT", version="1.1", clean_text=False, min_words=10, 
                 min_conf_score=0.20, max_length=256, top_k=1, pre_trained_name='bert-base-uncased'):
        
        self.lang_code = lang_code
        self.method = method
        self.version = version
        self.clean_text = clean_text
        self.min_words = min_words
        self.min_conf_score = min_conf_score
        self.max_length = max_length
        self.top_k = top_k
    
        self.model_path = abs_path + "/models"
        self.label_index_path = abs_path + "/label_index"
        subdir = "{}_{}_{}".format(self.lang_code, self.method, self.version)
        
        self.valid_langs = ["en"]
        if lang_code in self.valid_langs:
            self.filepath_model = self.model_path + "/" + subdir + "/model.pt"
            self.filepath_label = self.label_index_path + "/" + subdir + "/label_index.json"
        
            if os.path.isfile(self.filepath_model) and os.path.isfile(self.filepath_label):
                self.label_index = load_from_json(self.filepath_label)
                self.tokenizer = self.load_tokenizer(pre_trained_name)
                self.model = self.load_model(pre_trained_name, num_labels=len(self.label_index))
                self.model.to(device)
                self.model.load_state_dict(torch.load(self.filepath_model, device), strict=False)

    
    @lru_cache(maxsize=128)
    def load_model(self, pre_trained_name, num_labels):
        """ Load BERT pre-trained model with given num labels 
        
        :param pre_trained_name: BERT pre-trained model name
        :param num_labels: total number of classes
        :return: BertForSequenceClassification model
        """
        model = BertForSequenceClassification.from_pretrained(
            pre_trained_name, 
            num_labels=num_labels,
            output_attentions=False, 
            output_hidden_states=False
        )
        return model

    @lru_cache(maxsize=128)
    def load_tokenizer(self, pre_trained_name):
        """ Load BERT pre-trained tokenizer
        
        :param pre_trained_name: BERT pre-trained model name
        :return: BERT tokenizer
        """
        tokenizer = BertTokenizerFast.from_pretrained(pre_trained_name)
        return tokenizer 
    
    def predict_sentiment(self, text):
        """ Predict sentiment label for new documents 
        
        :param text: text 
        :return: python dictionary
        Sample
        {
            "label": "positive",
            "confidence": "0.896",
            "predictions":[
                {
                    "label":"positive",
                    "confidence":"0.896"
                },
                {
                    "label":"neutral",
                    "confidence":"0.054"
                },
                {
                    "label":"negative",
                    "confidence":"0.050"
                }
            ],
            "message":"successful"
        }
        """
        try:
            prediction = dict()

            if text:
                if Utility.get_doc_length(text) > self.min_words:
                    if self.lang_code in self.valid_langs:
                        if self.clean_text:
                            text = Utility.clean_text(text)

                        if self.method == "BERT":
                            if os.path.isfile(self.filepath_model) and os.path.isfile(self.filepath_label):
                                # tokenize, encode and generate input_ids, attention_mask
                                p = Preprocessing(self.tokenizer)
                                input_ids, attention_mask = p.text_preprocessing([text], max_length=self.max_length)
                                input_ids = input_ids.to(device)
                                attention_mask = attention_mask.to(device)
                                
                                # generate index_label dictionary and get the list of class names
                                index_label = {v: k for k, v in self.label_index.items()}
                                class_names = list(index_label.values())

                                predictions = []
                                with torch.no_grad():
                                    # add input_ids, attention_mask to BERT pre-trained model
                                    outputs = self.model(input_ids, attention_mask)                     # (tensor([[-0.0669, -0.7634, -1.2053,  1.7891, -0.8443]]),)
                                    tensors = outputs[0][0]                                             # tensor([-0.0669, -0.7634, -1.2053,  1.7891, -0.8443])
                                    top_k_preds = torch.topk(tensors, self.top_k)                   
                                    tensor_scores = top_k_preds[0]                                      # tensor([ 1.7891, -0.0669, -0.7634])
                                    tensor_indexes = top_k_preds[1]                                     # tensor([3, 0, 1])                      
                                    # convert tensors to probabilities
                                    confidences = F.softmax(tensor_scores, dim=0)                       # tensor([0.8103, 0.1266, 0.0631])
                                    confidences = confidences.tolist()                                  # [0.8102552890777588, 0.12663434445858002, 0.06311030685901642]
                                
                                    # loop through each confidence and get associated index label
                                    for index, confidence in enumerate(confidences):
                                        # get confidence tensor position
                                        tensor_index = top_k_preds[1][index].item() 
                                        label_pred = dict()
                                        label_pred['label'] = class_names[tensor_index]
                                        label_pred['confidence'] = "{0:.3f}".format(confidence)
                                        predictions.append(label_pred)
                                
                                if predictions:
                                    max_conf_label = max(predictions, key=lambda k: k["confidence"])
                                    label = max_conf_label.get("label")
                                    confidence = max_conf_label.get("confidence")
                                    
                                    if float(confidence) <= self.min_conf_score:
                                        return "unknown sentiment, confidence below threshold"
                            
                                    prediction["label"] = label
                                    prediction["confidence"] = confidence
                                    prediction["predictions"] = predictions
                                    prediction["message"] = "successful"
                                    return prediction
                                else:
                                    return "no labels found"
                            else:
                                return "model not found"
                        else:
                            return "method not found"
                    else:
                        return "language not supported"
                else:
                    return "required at least {} words for prediction".format(self.min_words)
            else:
                return "required textual content"
        except Exception:
            logging.error("exception occured", exc_info=True)


