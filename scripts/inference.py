from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats
import yaml
import os
from decolle.decolle import DECOLLE
import torch
import numpy as np
from decolle.utils import accuracy
from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import create_dataloader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_dir = 'logs/lenet_decolle_cuda/default/Feb18_12-31-32_iiticubws036/'
params_file = os.path.join(model_dir, 'params.yml')
checkpoint_dir = os.path.join(model_dir, 'checkpoints')

with open(params_file, 'r') as f:
    params = yaml.load(f)

net = DECOLLE( out_channels=params['out_channels'],
                    Nhid=params['Nhid'],
                    Mhid=params['Mhid'],
                    kernel_size=params['kernel_size'],
                    pool_size=params['pool_size'],
                    input_shape=params['input_shape'],
                    alpha=params['alpha'],
                    alpharp=params['alpharp'],
                    beta=params['beta'],
                    num_conv_layers=params['num_conv_layers'],
                    num_mlp_layers=params['num_mlp_layers'],
                    lc_ampl=params['lc_ampl']).to(device)

load_model_from_checkpoint(checkpoint_dir, net, None, device)
net.eval()

_, data_loader = create_dataloader(batch_size=1)

loss = torch.nn.SmoothL1Loss()

# test(data_loader, loss, net, 50)
for data, target in data_loader:
    predictions = []
    for i in range(data.shape[1]):
        spikes, readouts, mem_potentials = net(data[:, i, ...])
        predictions.append(np.array([torch.sigmoid(r).detach().numpy() for r in readouts]))
    acc = accuracy(np.swapaxes(predictions, 0, 1), target.detach().numpy())
    print(acc)
