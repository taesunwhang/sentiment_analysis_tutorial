### Sentiment Analysis Tutorial
Convolution Neural Networks for Sentiment Analysis (Tutorial)

### Prepare Dataset, Make Vocabulary and GLoVE Embeddings
The full dataset is publicly available in https://www.kaggle.com/bittlingmayer/amazonreviews  
The original GLoVE embedding is avaliable in  https://nlp.stanford.edu/projects/glove/   
To sample data from the full dataset, run
    
    python3 data_utils.py
    
### Training CNN for sentiment analysis 
    
    python3 main.py --model=cnn_classifier
   
### Examples of training output
    [Step 10] loss: 0.7052, accuracy: 56.88%
    [Step 20] loss: 0.7016, accuracy: 61.72%
    [Step 30] loss: 0.5723, accuracy: 69.06%
    [Step 40] loss: 0.5069, accuracy: 75.00%
    [Step 50] loss: 0.5501, accuracy: 72.50%


