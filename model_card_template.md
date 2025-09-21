# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model performs binary classification to predict whether an individual's annual income exceeds $50,000. 
It is implemented with scikit learn's LogisticRegression and is trained on one-hot encoded categorical features together with numeric fields prepared with preprocessing.
The codebase uses Python, pandas, NumPy, and scikit-learn. The repository is maintained at https://github.com/TBriggs-WGU/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.
The current version was produced on 9/21/2025. 

## Intended Use
The intended use of this model is educational. It demonstrates how to build a machine-learning workflow that includes training, evaluation on data slices, and serving with FastAPI.
It is not intended for any real-world decision making such as employment, lending, housing, or healthcare. 
Any deployment beyond the course would require extensive validation and governance. 

## Training Data
The model is trained on the UCI Adult (Census Income) datase, which is provided locally as data/census.csv.
The target column is salary with labels <=50k and >50k. 
During preprocessing, the function process_data fits a OneHotEncoder on the categorical features and a LabelBinarizer on the target.
The encoders are fit on the training split and reused for transforming the test split and any future inference data to avoid leakage.

## Evaluation Data
I split the dataset using training and test partitions using an 80/20 split that is stratified on the salary label with a fixed random seed of 42. 
In addition to overall metrics, I compute performance on data slices, where each slice holds one categorical feature value fixed (for example, a single value of education or workclass).
Those slice metrics are written to slice_output.txt to facilitate inspection for disparities.

## Metrics
I report precision, recall, and F1 score.
On  the held-out test set, the model achieved a precision of 0.7088, a recall of 0.2717, and an F1 score of 0.3928.
These results indicate that, with default settings, the model favors precision over recall and misses a substantial number of positive cases.
Improving recall may require class weighting, threshold tuning, or trying alternative models.

## Ethical Considerations
The dataset contains sensitive attributes such as sex, race, and native-country. Historical bias in the data can be reflected or amplified by the model.
Any operational use must include careful review of per-slice performance, fairness analysis, and mitigation steps.
The outputs are probalistic and should not be interpreted as definitive statements about individuals. 


## Caveats and Recommendations
This baseline was trained without hyperparameter tuning or probability calibration.
The metrics come from a single train/test split and may vary across splits. 
Cross-validation would provide more stable estimates.
If this model is extended, I recommend adding class weighting or threshold  tuning to improve recall, performing systematic hyperparameter searches, and monitoring for data drift over time. 
Any threshold or configuration changes should be documented along with their impact on precision and recall.  