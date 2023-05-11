import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import *
from Encoder import *
from FFN import *
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
# test = encode(encoder, test)

# labels to numpy 
test_labels = np.array(test_labels)

# Initialize the SVMClassifier
# saved_model = torch.load('model.pkl')
# encoder = Encoder(config['input_size'], config['hidden_size'])
# ffn = FFN(config['hidden_size'], config['hidden_size']//2, config['output_size'])

# mlp_classifier = Combined(encoder, ffn).to(config['device'])
# mlp_classifier.load_state_dict(saved_model)
# svm_classifier = pickle.load(open('svm_classifier.pkl', 'rb'))
# gb_classifier = pickle.load(open('gb_classifier.pkl', 'rb'))
xgb_classifier = pickle.load(open('xgb_classifier.pkl', 'rb'))

# # Make predictions on the test set
# mlp_pred = []
# for x in test:
#     # print(x)
#     # print(x.shape)
#     x = torch.tensor(x, dtype=torch.float32)
#     # print(type(x))
#     mlp_pred.append(mlp_classifier(x))
# mlp_pred = np.array(mlp_pred)
test = encode(encoder, test)
# svm_pred = svm_classifier.predict(test)
# gb_pred = gb_classifier.predict(test)
xgb_pred = xgb_classifier.predict(test)

# Save each model's predictions to a csv file
# with open('mlp_pred.csv', 'w') as f:
    # f.write('id, label\n')
#     for i, x in enumerate(mlp_pred):
#         f.write('patient_'+ str(i) + ', ' + str(x) + '\n')
# with open('svm_pred.csv', 'w') as f:
#     f.write('id, label\n')
#     for i, x in enumerate(svm_pred):
#         f.write('patient_'+ str(i) + ', ' + str(x) + '\n')
# with open('gb_pred.csv', 'w') as f:
#     f.write('id, label\n')
#     for i, x in enumerate(gb_pred):
#         f.write('patient_'+ str(i) + ', ' + str(x) + '\n')
with open('prediction.csv', 'w') as f:
    f.write('id, label\n')
    for i, x in enumerate(xgb_pred):
        f.write('patient_'+ str(i) + ', ' + str(x) + '\n')