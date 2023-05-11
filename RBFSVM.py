import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import *
from Encoder import Encoder
import pickle

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

# Initialize the SVMClassifier
svm_classifier = SVC(kernel='rbf', C=0.5)

# Train the classifier
svm_classifier.fit(train, labels)

# Make predictions on the test set
y_pred = svm_classifier.predict(test)

# Calculate the accuracy and F1 score
accuracy = accuracy_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
get_roc(test_labels, y_pred, 'SVM_test')

print('RBFSVM Test')
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Make predictions on the train set
y_pred_train = svm_classifier.predict(train)

# Calculate the accuracy and F1 score for train
accuracy = accuracy_score(labels, y_pred_train)
f1 = f1_score(labels, y_pred_train)
precision = precision_score(labels, y_pred_train)
recall = recall_score(labels, y_pred_train)
get_roc(labels, y_pred_train, 'SVM_train')

print('RBFSVM Train')
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

pickle.dump(svm_classifier, open('svm_classifier.pkl', 'wb'))
