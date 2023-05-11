import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import *
from Encoder import *
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import pickle

# Load the data
test, test_labels = get_test()

encoder = Encoder(config['input_size'], config['hidden_size'])
if os.path.exists('encoder.pkl'):
    encoder.load('encoder.pkl')
else:
    print('Encoder not found!')

# Encode the input data
test = encode(encoder, test)

# labels to numpy 
test_labels = np.array(test_labels)

# Initialize the SVMClassifier
mlp_classifier = torch.load('model.pkl')
svm_classifier = pickle.load(open('svm_classifier.pkl', 'rb'))
gb_classifier = pickle.load(open('gb_classifier.pkl', 'rb'))
xgb_classifier = pickle.load(open('xgb_classifier.pkl', 'rb'))

# Make predictions on the test set
mlp_pred = []
for x in test:
    x = torch.tensor(x, dtype=torch.float32)
    print(type(x))
    mlp_pred.append(mlp_classifier(x).item())
mlp_pred = np.array(mlp_pred)
svm_pred = svm_classifier.predict(test)
gb_pred = gb_classifier.predict(test)
xgb_pred = xgb_classifier.predict(test)

# Save each model's predictions to a csv file
with open('mlp_pred.csv', 'w') as f:
    f.write('id, label\n')
    for i, x in enumerate(mlp_pred):
        f.write('patient_'+ str(i) + ', ' + str(x) + '\n')
with open('svm_pred.csv', 'w') as f:
    f.write('id, label\n')
    for i, x in enumerate(svm_pred):
        f.write('patient_'+ str(i) + ', ' + str(x) + '\n')
with open('gb_pred.csv', 'w') as f:
    f.write('id, label\n')
    for i, x in enumerate(gb_pred):
        f.write('patient_'+ str(i) + ', ' + str(x) + '\n')
with open('xgb_pred.csv', 'w') as f:
    f.write('id, label\n')
    for i, x in enumerate(xgb_pred):
        f.write('patient_'+ str(i) + ', ' + str(x) + '\n')