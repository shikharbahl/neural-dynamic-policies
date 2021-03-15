import scipy.io as sio
import torch
from torch.autograd import Variable
import numpy as np


class Mapping:
    y_max = 1
    y_min = -1
    x_max = []
    x_min = []


class MatLoader:
    def load_data(file,
                  load_original_trajectories=False,
                  image_key='imageArray',
                  traj_key='trajArray',
                  dmp_params_key='DMPParamsArray',
                  dmp_traj_key='DMPTrajArray'):
        # Load data struct

        # file = Set file variable to /path/to/arr.m 
        data = sio.loadmat(file)

        # Parse data struct
        if 'Data' in data:
            data = data['Data']
        # Backward compatibility with old format
        elif 'slike' in data:
            data = data['slike']
            image_key = 'im'
            traj_key = 'trj'
            dmp_params_key = 'DMP_object'
            dmp_traj_key = 'DMP_trj'

        # Load images
        images = []
        for image in data[image_key][0]:
            images.append(image.astype('float'))
        images = np.array(images)

        # Load DMPs
        DMP_data = data[dmp_params_key][0]
        outputs = []
        for dmp in DMP_data:
            tau = dmp['tau'][0, 0][0, 0]
            w = dmp['w'][0, 0]
            goal = dmp['goal'][0, 0][0]
            y0 = dmp['y0'][0, 0][0]
            # dy0 = np.array([0,0])
            learn = np.append(tau, y0)
            # learn = np.append(learn,dy0)
            learn = np.append(learn, goal)  # Correction
            learn = np.append(learn, w)
            outputs.append(learn)
        outputs = np.array(outputs)
        '''
        scale = np.array([np.abs(outputs[:, i]).max() for i in range(0, 5)])
        scale = np.concatenate((scale, np.array([np.abs(outputs[:, 5:outputs.shape[1]]).max() for i in range(5, outputs.shape[1])])))
        '''

        # Scale outputs
        y_max = 1
        y_min = -1
        x_max = np.array([outputs[:, i].max() for i in range(0, 5)])
        x_max = np.concatenate(
            (x_max, np.array([outputs[:, 5:outputs.shape[1]].max() for i in range(5, outputs.shape[1])])))
        x_min = np.array([outputs[:, i].min() for i in range(0, 5)])
        x_min = np.concatenate(
            (x_min, np.array([outputs[:, 5:outputs.shape[1]].min() for i in range(5, outputs.shape[1])])))
        scale = x_max-x_min
        scale[np.where(scale == 0)] = 1
        outputs = (y_max - y_min) * (outputs-x_min) / scale + y_min

        # Load scaling
        scaling = Mapping()
        scaling.x_max = x_max
        scaling.x_min = x_min
        scaling.y_max = y_max
        scaling.y_min = y_min

        # Load original trajectories
        original_trj = []
        if load_original_trajectories:
            trj_data = data[traj_key][0]
            original_trj = [(trj) for trj in trj_data[:]]

        return images, outputs, scaling, original_trj

    def data_for_network(images, outputs):
        input_data = Variable(torch.from_numpy(images)).float()
        output_data = Variable(torch.from_numpy(outputs), requires_grad=False).float()
        return input_data, output_data
