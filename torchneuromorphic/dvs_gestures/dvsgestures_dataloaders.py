#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Author: Emre Neftci
#
# Creation Date : Fri 01 Dec 2017 10:05:17 PM PST
# Last Modified : Sun 29 Jul 2018 01:39:06 PM PDT
#
# Copyright : (c)
# Licence : Apache License, Version 2.0
#-----------------------------------------------------------------------------
import struct
import time
import numpy as np
import scipy.misc
import h5py
import torch.utils.data
from .create_hdf5 import create_events_hdf5
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from ..transforms import *
import os

mapping = { 0 :'Hand Clapping'  ,
            1 :'Right Hand Wave',
            2 :'Left Hand Wave' ,
            3 :'Right Arm CW'   ,
            4 :'Right Arm CCW'  ,
            5 :'Left Arm CW'    ,
            6 :'Left Arm CCW'   ,
            7 :'Arm Roll'       ,
            8 :'Air Drums'      ,
            9 :'Air Guitar'     ,
            10:'Other'}

class DVSGestureDataset(NeuromorphicDataset):
    resources_url = [['https://public.boxcloud.com/d/1/b1!edl9nIySCYJgdBLyQcRAkKngxghPdyMrAyk4j9Skf3pNkcBcQNcU3G8HhjQkOF5Y7vH6HkgeyFeq3DQ4TEVqoG9oZj15M7wIXph-1s4m6aS9O2vF8BYZrsYceTDKlBipg2LU-5GRTkRbOsb3eRA00YMFELBtiRZzDnneEtjjZUad1VSfpi9cWgB2uiHob3w5uu2hdoamHkoVOcf3AV0LCbbwfW26gLubcS9hiVIdrfMt0LxmQsbsrQpsEE2dYJTn7E2fKTSKQUhb65HHsugq0F6v0bNK5ZbllGRbcGGO43wKj5YCfsWO4TwisWvUO6pzjsUCk5HK2KyepC6gIFtl8f2PV9O3okV0mvoiF1TmDgGihMyqoxRwdIxEnZLpU3D2qMx3j2XH1QxkGXJQA09igiZoOVMTTxR6Be2dRHG2ZJdsBa0WE5e4ZlxpNzgAsebEBFFMnE6egcPAxokVhO0TpM1lv4PcVWaXJqYf8l-qz5Ofbj3ShKGIJO_V-PFvBtrg2KsqhJi2ma0fgTznEBIDjahgccweSBalieG9k2-7K7hNYa59kuXg9yL1N6I3abU5JR3cCnesiEAJX5cgS6MRIf_uptRQ870ijiDRGUO4eDp58jKZSJAjXQ_wvJCmZ-3Eam6MFyTuJ_XkmV0zL76kygnHdBeTTGQjpZB3dU-ClScx0sZS3AEQWc25SGuOA5Nk9U786UOSDqhM3_khlS_GozoxIz3t6_GtrZNBRE0m-yJpX0SlGudcJ6vYLlmgL4ylZjFZHwn2oVcqw1fY_pULjRL3MjStI0xQYRa-PeuvQ4tZQvR9YXZTlc38AE8OI-NQsnifew_GXGLOxLjJb5Mmcg-27e3SsbMfcgPBDIHCIEYcz5eCdV9cTV3Sa2HPnMF0fUIaf63t1f8D8N-TEWa6MhzhrmxZ1HCaYi2Lqle9WqbVGDurEBZ5jWJ4UE8cUgz6r-B2YkqXm7oJg1SJ0xysgTGw4WX5N7nLFRSSCuQhoQ5DvnbFVKjxhaVx-IimsIDTMTExNDGgi_4S8Sz9xPPc-XT7GSLaE6gk9fB5ij8klguxXHXIu4nsDFGdTRvvHd72A2vYsS2KMV0PzSxWxg_uCE4RCwF2DTU9UFnZyH-hU7JxoDk5VAplDiXFFd104SfyMlZpVD3uCMOKwcfkXdh9dFclRA5LWDeJLXRLb7Y1HPEyi-0d6jq1RUivSksWrG2gCagAXtIDAEVOIfQcrKrlTk_kd_UET8eI9XZgBTBuPJb45g3T1zQhnwlZX0wtv0s2s2XXYz7VGpi3pjZyRow8M0Vs_KD4416JpfGpOUwDdbckvOfxSy8DNWcpI2MvawFelB-_Zw../download',None, 'DvsGesture.tar.gz']]
    directory = 'data/dvsgesture/'
    resources_local = [directory + 'DvsGesture']

    def __init__(
            self, 
            root,
            train=True,
            transform=None,
            target_transform=None,
            download_and_create=True,
            chunk_size = 500):

        self.n = 0
        self.download_and_create = download_and_create
        self.root = root
        self.train = train 
        self.chunk_size = chunk_size

        super(DVSGestureDataset, self).__init__(
                root,
                transform=transform,
                target_transform=target_transform )

        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            if train:
                self.n = f['extra'].attrs['Ntrain']
                self.keys = f['extra']['train_keys']
            else:
                self.n = f['extra'].attrs['Ntest']
                self.keys = f['extra']['test_keys']

    def download(self):
        super(DVSGestureDataset, self).download()

    def create_hdf5(self):
        create_events_hdf5(self.resources_local[0], self.root)

    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
            if not self.train:
                key = key + f['extra'].attrs['Ntrain']
            data, target = sample(
                    f,
                    key,
                    T = self.chunk_size,
                    shuffle=self.train)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

def sample(hdf5_file,
        key,
        T = 500,
        shuffle = False):
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    tbegin = np.maximum(0,dset['times'][0]- 2*T*1000)
    tend = dset['times'][-1] 
    start_time = np.random.randint(tbegin, tend) if shuffle else 0

    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad[:,0]-=tmad[0,0]
    return tmad[:, [0,3,1,2]], label

 
def create_dataloader(
        root = 'data/dvsgesture/dvs_gestures_build19.hdf5',
        batch_size = 72 ,
        chunk_size_train = 500,
        chunk_size_test = 1800,
        ds = None,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        **dl_kwargs):
    if ds is None:
        ds = 4
    size = [2, 128//ds, 128//ds]

    if transform_train is None:
        transform_train = Compose([
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_train, size = size),
            ToTensor()])
    if transform_test is None:
        transform_test = Compose([
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_test, size = size),
            ToTensor()])
    if target_transform_train is None:
        target_transform_train =Compose([Repeat(chunk_size_train), toOneHot(11)])
    if target_transform_test is None:
        target_transform_test = Compose([Repeat(chunk_size_test), toOneHot(11)])

    train_d = DVSGestureDataset(root,
                                train=True,
                                transform = transform_train, 
                                target_transform = target_transform_train, 
                                chunk_size = chunk_size_train)

    train_dl = torch.utils.data.DataLoader(train_d, batch_size=batch_size, **dl_kwargs)

    test_d = DVSGestureDataset(root,
                               transform = transform_test, 
                               target_transform = target_transform_test, 
                               train=False,
                               chunk_size = chunk_size_test)

    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl



