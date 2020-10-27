# BERT-Sentiment-Analysis

BERT-Sentiment-Analysis is an NLP task meant to help in identifying and understanding user opinion as positive, neutral, or negative with respect to a given topic. It has been developed using Google Play reviews data for generating classification model using state-of-the-art BERT pre-trained model.
 
Technologies used: Python3, BERT, Jupyter Notebook, matplotlib, seaborn, Google Colab
<div align="center"><img width="793" alt="Screenshot 2020-10-26 at 15 10 39" src="https://user-images.githubusercontent.com/11573356/97183256-da050b00-179d-11eb-9f93-415289444a08.png"></div>



## Project Outline:
```
  - Data Acquisition
  - Generating ground-truth label dataset
  - Model Training & Evaluation
  - Sentiment prediction

Basic project installation steps:

  1. Clone repository

  2. Generate model & evaluation files:
     - load dataframe                                 : required dataset with "label", "text" colnames
     - check sequence length distribution in dataset  : required for BERT pre-trained model (max length 512 tokens)
     - use power of 2 to determine max_length           (e.g 128, 256, 512)
     - import and create Evaluation object
     - create model using create_model() function

          from evaluation import Evaluation
          df = pd.read_csv('data/google_play_reviews/dataset.csv')
          ev = Evaluation(lang_code="en", method="BERT", version="1.1", epochs=10)
          ev.create_model(df=df, max_length=max_length, output_path="output")

     Evaluation files:
        - data distribution: sequence length, original, train, test datasets
        - plot train history
        - plot confusion matrix
        - classification report
        - label_index json file                        : label-index mapping
            {
              "negative": 0,
              "neutral": 1,
              "positive": 2
            }

  3. Predict sentiment for new documents:
      - import and create Sentiment object
      - predict sentiment using predict_sentiment() function

         from sentiment import Sentiment
         s = Sentiment(lang_code="en", method="BERT", version="1.1")
         pred = s.predict_sentiment("text_to_predict)

   Sample:
         text = "Being happy doesn't mean you'll live longer. I am sad about this life that is too short!"
         s = Sentiment(lang_code="en", method="BERT", version="1.1")
         pred = s.predict_sentiment(text)
         print(pred)
         '''
             {
                 "label":"neutral",
                 "confidence":"0.847",
                 "predictions":[
                      {
                         "label":"neutral",
                         "confidence":"0.847"
                      },
                      {
                         "label":"positive",
                         "confidence":"0.149"
                      },
                      {
                         "label":"negative",
                         "confidence":"0.003"
                      }
                 ],
                 "message":"successful"
             }
          '''
       
```

## Classification report:
```
              precision    recall  f1-score   support

    negative       0.84      0.80      0.82       769
     neutral       0.72      0.71      0.72       758
    positive       0.84      0.88      0.86       903

    accuracy                           0.80      2430
   macro avg       0.80      0.80      0.80      2430
weighted avg       0.80      0.80      0.80      2430
```
