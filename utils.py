import pandas
import os
import numpy as np
import torch
from collections import OrderedDict
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
# import wandb
# import os
from tqdm import tqdm


# Set up wandb environment
# os.environ["WANDB_PROJECT"] = "LAB2-EX1"
# os.environ["WANDB_LOG_MODEL"] = "true"

config = {
    'epochs': 10,
    'batch_size': 1,
    'learning_rate': 0.001,
    'input_size': 26,
    'hidden_size': 100,
    'output_size': 1,
    'train_size': 0.8,
    'val_size': 0.2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dropset': ['Alkalinephos', 'PaCO2', 'Bilirubin_total', 'TroponinI', 'Fibrinogen', 'FiO2', 'EtCO2', 'AST', 'PTT', 'Bilirubin_direct', 'pH', 'BaseExcess', 'Lactate', 'SaO2']
}


def read_psv_files_to_dict(path):
    """Reads all .psv files in a directory and returns a dictionary of
    pandas dataframes.
    """
    data = dict()
    for filename in os.listdir(path):
        if filename.endswith('.psv'):
            name = filename.split('.')[0]
            data[name] = pandas.read_csv(os.path.join(path, filename),
                                         sep='|')
    data = OrderedDict(sorted(data.items(), key=lambda x: int(x[0].split('_')[1])))

    return data

def get_pandas():
    data = read_psv_files_to_dict('data/train')
    for key, pat in data.items():
        data[key] = pat.mean(axis=0)
    df = pandas.DataFrame.from_dict(data)
    df = df.transpose()
    return df

def get_input_for_model(data, dropset = None):
    res, targets = OrderedDict(), []
    if dropset is None:
        dropset = count_null_columns(data)
        print('only null cols are: ', dropset)
    for patient in data.keys():
        df = data[patient]
        if not df[df['SepsisLabel'] == 1].empty:
            sepsis_index = df[df['SepsisLabel'] == 1].index[0]
            selected_rows = df.loc[:sepsis_index]
        else:
            sepsis_index = -1
            selected_rows = df
        input_data = selected_rows.drop(['SepsisLabel'], axis=1)
        for col in dropset:
            input_data.drop(col, axis=1, inplace=True)
        targets.append(0 if sepsis_index == -1 else 1)
        
        for column in input_data.columns:
                input_data[column].interpolate(method='linear', inplace=True)
        input_data.fillna(-11, inplace=True)
                
        res[patient] = input_data
    return res, targets


def tensorize_input(inputs, targets):
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    inputs_tensor = []
    for patient in inputs.keys():
        np_arr = inputs[patient].values
        inputs_tensor.append(torch.tensor(np_arr, dtype=torch.float32))
    return inputs_tensor, targets_tensor

def count_null_columns(dataframes):
    onlynull = set()
    for _, df in dataframes.items():
        all_null_columns = df.isnull().all()
        only_null_columns = [col for col in all_null_columns.index if all_null_columns[col]]
        onlynull = onlynull.union(only_null_columns)
        return onlynull
    
def check_if_none(dataframes):
    for _, df in dataframes.items():
        if df.isnull().all().any():
            return True
    return False

def get_train():
    if not os.path.exists('inputs.pkl') or not os.path.exists('targets.pkl'):
        data = read_psv_files_to_dict('data/train')
        inputs, targets = get_input_for_model(data)
        inputs, targets = tensorize_input(inputs, targets)
        torch.save(inputs, 'inputs.pkl')
        torch.save(targets, 'targets.pkl')
    else:
        inputs = torch.load('inputs.pkl')
        targets = torch.load('targets.pkl')
    return inputs, targets

def encode(encoder, inputs):
    encoded = []
    for input in inputs:
        encoded.append(encoder(input).detach().numpy())
    return np.array(encoded).squeeze()

def to_numpy(inputs):
    res = []
    for input in inputs:
        res.append(input.detach().numpy())
    return np.array(res).squeeze()

def get_roc(y_true, y_pred_prob, name):
    # Assuming you have true labels (y_true) and predicted probabilities (y_pred_prob) from your model
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(name + '_roc.png')

def get_test():
    if not os.path.exists('test_inputs.pkl') or not os.path.exists('test_targets.pkl'):
        data = read_psv_files_to_dict('data/test')
        test_inputs, test_targets = get_input_for_model(data, dropset = config['dropset'])
        test_inputs, test_targets = tensorize_input(test_inputs, test_targets)
        torch.save(test_inputs, 'test_inputs.pkl')
        torch.save(test_targets, 'test_targets.pkl')
    else:
        test_inputs = torch.load('test_inputs.pkl')
        test_targets = torch.load('test_targets.pkl')
    return test_inputs, test_targets

get_pandas()