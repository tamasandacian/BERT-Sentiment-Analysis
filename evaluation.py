from visualization import save_train_history, save_confusion_matrix, save_class_distribution, save_seq_len_distribution
from collections import defaultdict
from sklearn.metrics import classification_report
from shared.utils import dump_to_json
from shared.utils import dump_to_txt
from shared.utils import make_dirs
from cm import CM
import numpy as np
import torch

class Evaluation(object):
    """ Class for generating classification model and evaluation files.
    """
    def __init__(self, lang_code="en", method="BERT", version="1.1", pre_trained_name='bert-base-uncased',
                 epochs=4, batch_size=32, test_size=0.15, random_state=42, lr=2e-5, eps=1e-8, text_col="text", 
                 label_col="label"):
        
        self.lang_code = lang_code
        self.method = method
        self.version = version
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.lr = lr
        self.eps = eps
        self.pre_trained_name = pre_trained_name
        self.text_col = text_col
        self.label_col = label_col

    def actual_labels(self, test_y, index_label):
        """ Convert numeric actual labels to categorical format 
        
        :param test_y: numeric numpy labels
        :param index_label: index-label mapping
        :return: list of strings
        """
        actual_labels = list(test_y.numpy())
        actual_labels = [index_label[label] for label in actual_labels]
        actual_labels
        return actual_labels

    def predicted_labels(self, preds, index_label):
        """ Convert numeric predicted labels to categorical format 
        
        :param preds: numeric numpy labels
        :param index_label: index-label mapping
        :return: list of strings
        """
        pred_labels = list(preds)
        pred_labels = [index_label[label] for label in pred_labels]
        pred_labels
        return pred_labels
            
    def create_model(self, df, max_length, output_path):
        """ Create & save model to a given output path 
        
        :param df: DataFrame
        :param max_length: max input length for training model
        :param output_path: path to save model, dictionary, corpus, evaluation files
        """
    
        # define output path
        subdir = "{}_{}_{}".format(self.lang_code, self.method, self.version)
        models_path = output_path + "/models/" + subdir
        eval_path = output_path + "/evaluation/" + subdir
        label_index_path = output_path + "/label_index/" + subdir

        # create directories
        make_dirs(output_path)
        make_dirs(models_path)
        make_dirs(eval_path)
        make_dirs(label_index_path)

        # create CM object and fit dataframe
        cm = CM(
            method=self.method, epochs=self.epochs, batch_size=self.batch_size, random_state=self.random_state, 
            lr=self.lr, eps=self.eps, pre_trained_name=self.pre_trained_name, text_col=self.text_col, 
            label_col=self.label_col, test_size=self.test_size
        )

        cm.fit(df, max_length)
       
        # loop through all epochs and generate best model with high accuracy
        history = defaultdict(list)
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1} / {self.epochs}')
            print("-" * 10)
            train_acc, train_loss = self.train(
                model=cm.model, 
                dataloader=cm.dataset.train_dataloader, 
                optimizer=cm.optimizer, 
                scheduler=cm.scheduler,
                device=cm.device, 
                num_samples=len(cm.dataset.train_text)
            )
            print(f'Train loss {train_loss} accuracy {train_acc}')
            
            val_acc, val_loss = self.evaluate(
                model=cm.model, 
                dataloader=cm.dataset.test_dataloader, 
                optimizer=cm.optimizer, 
                scheduler=cm.scheduler, 
                device=cm.device, 
                num_samples=len(cm.dataset.test_text)
            )
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print()
            
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            if val_acc > best_accuracy:
                torch.save(cm.model.state_dict(), models_path + '/model.pt')
                best_accuracy = val_acc
        
        with torch.no_grad():
            preds = cm.model(cm.dataset.test_seq.to(cm.device), cm.dataset.test_mask.to(cm.device))
            preds = preds[0]
            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis = 1)
 
        # save label desc to json file
        dump_to_json(cm.dataset.label_index, label_index_path + "/label_index.json", sort_keys=False)

        # generate and save classification report
        report = classification_report(cm.dataset.test_y, preds, target_names=cm.dataset.class_names)
        dump_to_txt(report, eval_path + "/classification_report.txt")

        # convert labels from numerical to categorical format
        actual_labels = self.actual_labels(cm.dataset.test_y, cm.dataset.index_label)
        pred_labels = self.predicted_labels(preds, cm.dataset.index_label)

        # save distribution for: sequence length, original, train & test datasets
        save_seq_len_distribution(df, eval_path + '/seq_length.png')
        save_class_distribution(df, eval_path + '/data_dist.png')
        save_class_distribution(cm.dataset.X, eval_path + '/train_dist.png')
        save_class_distribution(cm.dataset.Y, eval_path + '/test_dist.png')
        
        # save train history, confusion matrix
        save_train_history(history, eval_path + "/train_history.png")
        save_confusion_matrix(actual_labels, pred_labels, eval_path + '/confusion_matrix.png')

    
    def train(self, model, dataloader, optimizer, scheduler, device, num_samples):
        """ Train model using train dataloader with total number of documents in train set 

        :param model: BERT pre-trained model
        :param dataloader: generated torch train dataloader
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param num_samples: total number of train samples
        :return: total correct predictions, average loss
        """
        model = model.train()
        losses = []
        correct_predictions = 0.0
        for step, batch in enumerate(dataloader):
            # clear previously calculated gradients 
            model.zero_grad()  
            
            batch = [r.to(device) for r in batch]
            input_ids, attention_mask, labels = batch
            
            inputs = {
            'input_ids':        input_ids,
            'attention_mask':   attention_mask,
            'labels':           labels
            }
            
            # get model predictions for the current batch
            outputs = model(**inputs)
            loss = outputs[0]
            tensors = outputs[1]
            
            _, preds = torch.max(tensors, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            # backward pass to calculate the gradients
            loss.backward()
            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        return correct_predictions.double() / num_samples, np.mean(losses)

    def evaluate(self, model, dataloader, optimizer, scheduler, device, num_samples):
        """ Evaluate model using validation dataloader with total number of documents in validation set 

        :param model: BERT pre-trained model
        :param dataloader: generated torch train dataloader
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param num_samples: total number of train samples
        :return: total correct predictions, average loss
        """
        model = model.eval()
        losses = []
        correct_predictions = 0
        for step, batch in enumerate(dataloader):
            batch = [r.to(device) for r in batch]
            input_ids, attention_mask, labels = batch
            
            inputs = {
                'input_ids':        input_ids,
                'attention_mask':   attention_mask,
                'labels':           labels
            }
            
            # deactivate autograd
            with torch.no_grad():
                # get model predictions for the current batch
                outputs = model(**inputs)

            loss = outputs[0]
            tensors = outputs[1]

            _, preds = torch.max(tensors, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
        return correct_predictions.double() / num_samples, np.mean(losses)

        