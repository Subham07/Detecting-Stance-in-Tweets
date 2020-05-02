# Detecting-Stance-in-Tweets
Training different models and checking performance for detecting stance from dataset of Tweets

### Information about dataset:

**dataset_train.csv** : contains tweets with respect to given target and the Stance labels for that tweet- For, Against or None (Neutral) <br/>
**dataset_test.csv** : contains tweets with respect to given target without any Stance labels <br/>
**dataset_test_answers.csv** : contains the stance labels with respect to tweets as in the test file <br/>


### Python Files and algorithms implemented

**DetectingStanceWithBERT.ipynb** : Bi-directional Encoder Represenations from Transformers is used for training the data and predicting the test data. The program was run in Google Colab with GPU as runtime

**Detecting_Stance_CNN.ipynb.ipynb** : CNN has been used for learning the data after pre-processing and forming embedded matrix for each tweet using the pre-trained Word2Vec model. The program was run in Google Colab with CPU as runtime

**detecting_stance_svm.ipynb** : Here SVM classifier is used to train and predict data before Tf-IDF vectorizer was used to vectorize each word present in vocabulary. The program was run in Jupyter Notebook.

For each of the above approaches, classifiers were trained seperately for each Target.

Performance with **BERT** was the best among others producing highest F1 Score of **0.63** for Target "Atheism"


