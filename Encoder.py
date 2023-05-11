import torch
from tqdm import tqdm
from utils import *

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        return self.lstm(x)[1][0]
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
    
# data = read_psv_files_to_dict('data/minitrain')
# inputs, targets = get_input_for_model(data)
# inputs, targets = tensorize_input(inputs, targets)

# emb = Encoder(config['input_size'], config['hidden_size'])
