import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import *
from Encoder import Encoder

# Load the data
train, labels = get_train()
test, test_labels = get_test()

encoder = Encoder(config['input_size'], config['hidden_size'])
if os.path.exists('encoder.pkl'):
    encoder.load('encoder.pkl')
else:
    print('Encoder not found!')

# Encode the input data
train = encode(encoder, train)
test = encode(encoder, test)

# labels to numpy 
labels = np.array(labels)
test_labels = np.array(test_labels)

# Initialize the GradientBoostingClassifier
gb_classifier = GradientBoostingClassifier()
xgb_classifier = XGBClassifier()

# Train the classifier
gb_classifier.fit(train, labels)
xgb_classifier.fit(train, labels)

# Make predictions on the test set
y_pred = gb_classifier.predict(test)
y_pred_xgb = xgb_classifier.predict(test)

# Make predictions on the train set
y_pred_train = gb_classifier.predict(train)
y_pred_train_xgb = xgb_classifier.predict(train)

# Calculate the accuracy and F1 score
accuracy = accuracy_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
get_roc(test_labels, y_pred, 'GBM_test')

print('Gradient Boosting Classifier Test')
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Calculate the accuracy and F1 score for train
accuracy = accuracy_score(labels, y_pred_train)
f1 = f1_score(labels, y_pred_train)
precision = precision_score(labels, y_pred_train)
recall = recall_score(labels, y_pred_train)
get_roc(labels, y_pred_train, 'GBM_train')

print('Gradient Boosting Classifier Train')
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Calculate the accuracy and F1 score
accuracy = accuracy_score(test_labels, y_pred_xgb)
f1 = f1_score(test_labels, y_pred_xgb)
precision = precision_score(test_labels, y_pred_xgb)
recall = recall_score(test_labels, y_pred_xgb)
get_roc(test_labels, y_pred_xgb, 'XGB_test')

print('XGB Test')
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Calculate the accuracy and F1 score for train
accuracy = accuracy_score(labels, y_pred_train_xgb)
f1 = f1_score(labels, y_pred_train_xgb)
precision = precision_score(labels, y_pred_train_xgb)
recall = recall_score(labels, y_pred_train_xgb)
get_roc(labels, y_pred_train_xgb, 'XGB_train')

print('XGB Train')
print(f"Accuracy: {accuracy:.2f}")
print(f'F1 Score: {f1:.2f}')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")