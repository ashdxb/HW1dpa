import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Encoder import Encoder
from utils import *
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

class FFN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.l2(self.relu(self.l1(x))))
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class Combined(torch.nn.Module):
    def __init__(self, encoder, ffn):
        super(Combined, self).__init__()
        self.encoder = encoder
        self.ffn = ffn
    
    def forward(self, x):
        enc = self.encoder(x)
        return self.ffn(enc)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

inputs, targets = get_train()
test_inputs, test_targets = get_test()

encoder = Encoder(config['input_size'], config['hidden_size'])
if os.path.exists('encoder.pkl'):
    encoder.load('encoder.pkl')
ffn = FFN(config['hidden_size'], config['hidden_size']//2, config['output_size'])
model = Combined(encoder, ffn).to(config['device'])

def train_model(model, inputs, targets, test_inputs, test_targets, lr=config['learning_rate'], num_epochs=config['epochs'], batch_size=config['batch_size']):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_batches = len(inputs) // batch_size
    loss_fn = nn.BCELoss().to(config['device'])
    best_f1 = 0.5

    print('Training model...')
    for epoch in tqdm(range(num_epochs)):
        for batch, (x, y) in enumerate(zip(inputs,targets)):
            x, y = x.to(config['device']), y.to(config['device'])
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y.view(-1,1))
            loss.backward()
            optimizer.step()

        # Compute the F1 score on the test set at the end of epoch
        test_f1, _, _ = test_model(model, test_inputs, test_targets)
        print('F1 score for Epoch {} on test set: {}'.format(epoch, test_f1))
        if best_f1 < test_f1:
            best_f1 = test_f1
            model.save('model.pkl')
            encoder.save('encoder.pkl')

def test_model(model, inputs, targets, name = 'train'):
    model.eval()
    with torch.no_grad():
        y_pred = []
        for x, y in zip(inputs, targets):
            x, y = x.to(config['device']), y.to(config['device'])
            output = model(x)
            y_pred.append(output.item())
        y_pred = np.array(y_pred)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        accuracy = accuracy_score(targets, y_pred)
        f1 = f1_score(targets, y_pred)
        precision = precision_score(targets, y_pred)
        recall = recall_score(targets, y_pred)
        get_roc(targets, y_pred, f'FFN_{name}')
    return f1, precision, recall, accuracy

train = False
if os.path.exists('model.pkl') and not train:
    model.load('model.pkl')
else:
    train_model(model, inputs, targets, test_inputs, test_targets)

f1, precision, recall, accuracy = test_model(model, test_inputs, test_targets, 'test')
print('F1 score on test set: {}'.format(f1))
print('Precision score on test set: {}'.format(precision))
print('Recall score on test set: {}'.format(recall))
print('Accuracy score on test set: {}'.format(accuracy))

f1, precision, recall, accuracy = test_model(model, inputs, targets)
print('F1 score on train set: {}'.format(f1))
print('Precision score on train set: {}'.format(precision))
print('Recall score on train set: {}'.format(recall))
print('Accuracy score on train set: {}'.format(accuracy))