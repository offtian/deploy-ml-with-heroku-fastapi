# Model Card

This model card describes about the income classification model trained on census income dataset.

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The Scikit-Learn's Random Forest Classifier model is trained on the census income dataset. 

## Intended Use
The model is created to predict a person's income level (either >50k or <=50k) based on just a few characteristics of the person including demographics or occupation information.  

## Training Data
The dataset contains 30,162 cleaned salary level entries with 14 features, 80% are randomly chosen to train a model.

## Evaluation Data
The dataset contains 30,162 cleaned salary level entries with 14 features, 20% are randomly chosen to evaluate a model's performance.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The model is evaluated on precision, recall and f1 scores.
The model performance is displayed as follows:
- train_precision: 0.833, train_recall: 0.682, train_fbeta: 0.750
- test_precision: 0.783, test_recall: 0.622, test_fbeta: 0.693

## Ethical Considerations
The model is trained on a public dataset from UCI ML repository.

## Caveats and Recommendations
The model is just created for a demonstration purpose so it's not optimized for performance.