# Sentiment Classification
Text sentiment classifier to detect if the input phrase is positive or negative sentiment

## Implementation
In this project I have worked on predicting the sentiment of the input sentence using multiple approaches.

## Approach
The input text is pre processed. In the preprocessing phase:
* " 's " is replaced by "is" (Eg. It's -> It is)
* " # " is removed (Eg. #iPhone -> iPhone)
* " @ " is removed and alphabets are retained (Eg @user -> user)
* Any other non alphabet characters are removed
* http links are removed 
* Text is converted to lower case
* Contraction sentences are fixed (Eg. I've -> I have ; I'd -> I had)
* Words are lemmatized 

Used Logistic Regression, Naive Bayes and LSTM for modelling.<br>
Created a streamlit webapp using LSTM for prediction.<br><br>

Dataset distribution
![image](https://github.com/Surbhit01/SentimentClassification/assets/24591039/6ac48b66-71dd-467d-bd18-36679aeae445)

Common (top 20) positive sentiment words
![image](https://github.com/Surbhit01/SentimentClassification/assets/24591039/5cf7b41f-a6ac-4cbd-9b9b-60930afae686)

Common (top 20) negative sentiment words
![image](https://github.com/Surbhit01/SentimentClassification/assets/24591039/fd5384d4-7278-4da0-aed9-d01128323842)

Common (top 20) neutral sentiment words
![image](https://github.com/Surbhit01/SentimentClassification/assets/24591039/4d774745-fc83-485a-9fbd-79d5b5c92a60)

**From the above visualizations it is visible that there are certain words which are present in all the 3 types of sentiments. We'll remove the common words and visualize again.** <br>


Common (top 20) unique positive sentiment words
![image](https://github.com/Surbhit01/SentimentClassification/assets/24591039/c723953c-b9bb-4769-86e6-b31d5ec053e5)

Common (top 20) unique negative sentiment words
![image](https://github.com/Surbhit01/SentimentClassification/assets/24591039/1cc61c6f-186a-431e-8917-824c0378321e)

Common (top 20) unique neutral sentiment words
![image](https://github.com/Surbhit01/SentimentClassification/assets/24591039/b1a28ce3-06bf-4a6d-a1f8-ecff423fba47)

### Sample Predictions

**Positive**
![SentimentClassification_positive](https://github.com/Surbhit01/SentimentClassification/assets/24591039/231535a8-97e5-446f-af47-4f78686b97a8)

**Negative**
![SentimentClassification_negative](https://github.com/Surbhit01/SentimentClassification/assets/24591039/b250ced3-bb25-4927-842c-86bf106c8631)


