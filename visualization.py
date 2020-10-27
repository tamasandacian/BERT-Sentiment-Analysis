import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['figure.figsize'] = (10, 5)

def save_confusion_matrix(actual_labels, pred_labels, output_path):
    """ Save plot confusion matrix using actual, predicted labels to a given output path  
        
    :param actual_labels: list
    :param pred_labels: list
    :param output_path: output path
    """
    df = pd.DataFrame({'actual_labels': actual_labels, 'pred_labels': pred_labels}, columns=['actual_labels', 'pred_labels'])
    cm_df = pd.crosstab(df['actual_labels'], df['pred_labels'], rownames=['Actual'], colnames=['Predicted'], margins=True)
    plt.figure()
    ax = sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.savefig(output_path)
    plt.clf()
    
def save_train_history(history, output_path):
    """ Save plot train history  
        
    :param history: dictionary
    :param output_path: output path
    """
    plt.figure()
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='val accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(output_path)
    plt.clf()

def save_class_distribution(df, output_path):
    """ Save plot class distribution found in a dataset
        
    :param df: DataFrame
    :param output_path: output path
    """
    plt.figure()
    ax = sns.countplot(df['label'])
    plt.xlabel('label')
    plt.savefig(output_path)
    plt.clf()

def save_seq_len_distribution(df, output_path):
    """ Save plot text length distribution found in a dataset
        
    :param df: DataFrame
    :param output_path: output path
    """
    plt.figure()
    sentences = df['text'].tolist()
    seq_len = [len(sentence.split()) for sentence in sentences]
    pd.Series(seq_len).hist(bins=30)
    plt.savefig(output_path)
    plt.clf()
