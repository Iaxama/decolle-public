from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats
import yaml
import os
from decolle.decolle import DECOLLE
import torch
from collections import Counter
import numpy as np
from decolle.utils import accuracy
from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import create_dataloader
import yarp
import matplotlib.pyplot as plt
from skimage.transform import resize
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_yarp_buf():
    input_buf_array = bytearray(np.zeros((input_img_height, input_img_width), dtype=np.uint8))
    input_buf_image = yarp.ImageMono()
    input_buf_image.resize(input_img_width, input_img_height)
    input_buf_image.setExternal(input_buf_array, input_img_width, input_img_height)
    return input_buf_array, input_buf_image

if __name__ == '__main__':
    model_dir = 'logs/lenet_decolle_cuda/default/Feb19_11-47-02_iiticubws036/'
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

    # _, data_loader = create_dataloader(root='', batch_size=1)
    # Initialise YARP
    yarp.Network.init()
    # Create a port and connect it to the iCub simulator virtual camera
    input_port_pos = yarp.BufferedPortImageMono()
    input_port_pos.open("/decolle/pos/spikes:i")
    input_port_neg = yarp.BufferedPortImageMono()
    input_port_neg.open("/decolle/neg/spikes:i")

    yarp.Network.connect("/spiker/neg/image:o", "/decolle/neg/spikes:i")
    yarp.Network.connect("/spiker/pos/image:o", "/decolle/pos/spikes:i")

    im = None

    input_img_height = 128
    input_img_width = 128

    #   Input buffer initialization
    in_arr_pos, in_img_pos = init_yarp_buf()
    in_arr_neg, in_img_neg = init_yarp_buf()


    predictions = []
    loss = torch.nn.SmoothL1Loss()

    # _, data_loader = create_dataloader(root='/datasets/dvs_gestures_build19.hdf5', batch_size=1)
    # # test(data_loader, loss, net, 50)
    # for data, target in data_loader:
    #     for i in range(data.shape[1]):
    #         spikes, readouts, mem_potentials = net(data[:, i, ...])
    #         predictions.append(np.array([torch.sigmoid(r).detach().numpy() for r in readouts]))
    #     acc = accuracy(np.swapaxes(predictions, 0, 1), target.detach().numpy())
    #     print(acc)

    while True:
        now = time.time()
        # print(input_port_pos.getPendingReads())
        # print(input_port_neg.getPendingReads())
        input_img_pos = input_port_pos.read()
        input_img_neg = input_port_neg.read()
        if time.time() - now > 0.5:
            if not predictions:
                continue
            print(len(predictions))
            predictions = np.array(predictions)
            maxs = predictions.argmax(axis=-1)
            most_common_out = []
            for i in range(maxs.shape[1]):
                most_common_out.append(Counter(np.squeeze(maxs[:, i])).most_common()[0])

            predictions = []
            print(most_common_out)

        in_img_pos.copy(input_img_pos)
        in_img_neg.copy(input_img_neg)
        #   run detection/segmentation on frame
        spikes_pos = np.ascontiguousarray(in_arr_pos).reshape(input_img_height, input_img_width)
        spikes_neg = np.ascontiguousarray(in_arr_neg).reshape(input_img_height, input_img_width)

        spikes_pos = resize(spikes_pos, (32, 32))
        spikes_neg = resize(spikes_neg, (32, 32))

        in_tensor = torch.tensor(np.expand_dims(np.stack((spikes_pos, spikes_neg)), 0).astype(np.float32))

        spikes, readouts, mem_potentials = net(in_tensor)
        predictions.append(np.array([torch.sigmoid(r).detach().numpy() for r in readouts]))

        # yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
        # input_port.read(yarp_image)
        # display the image that has been read
        # if im is None:
        #     im = plt.imshow(frame)
        # else:
        #     im.set_data(frame)
        # plt.draw()
        # plt.pause(0.001)
        # Cleanup
    input_port_pos.close()

