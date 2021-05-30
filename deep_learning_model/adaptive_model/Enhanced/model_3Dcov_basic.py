import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Model_3D(nn.Module):

    def __init__(self, N=60, K=32, Tx=8, Channel=2):
        super(Model_3D, self).__init__()

        self.bn0 = nn.BatchNorm1d(2)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64,
                               kernel_size=(3), stride=(3), padding=(1))
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=256,
                               kernel_size=(3), stride=(1), padding=(1))
        self.bn2 = nn.BatchNorm1d(256)

        # predict narrow beam
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.2)
        self.fc1 = nn.Linear(256, 64)
        # predict wide beam
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.2)
        self.fc2 = nn.Linear(256, 16)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)


    def forward(self, x, device):

        if 0:
            # MPC
            # trained wide beam number
            k = 7
            # batch size
            batch_size = 16
            # save narrow beam prediction results
            y1 = torch.zeros((10, 16, 64)).to(device)
            # save wide beam prediction results
            y2 = torch.zeros((10, 16, 16)).to(device)

            # first loop for wide beam training numbers
            for i in range(10):

                # if i = 0, full wide beam training
                if i == 0:
                    x_test = x[:, :, i, :]

                    #CNN
                    x_test = self.bn0(x_test)
                    x_test = self.conv1(x_test)
                    x_test = self.bn1(x_test)
                    x_test = F.relu(x_test)
                    x_test = self.conv2(x_test)
                    x_test = self.bn2(x_test)
                    x_test = F.relu(x_test)
                    P_dim_size = x_test.shape[2]
                    x_test = nn.MaxPool1d(kernel_size=P_dim_size)(x_test)

                    x_test = x_test.permute(2, 0, 1)
                    # predict narrow beam
                    y_test, (hn, cn) = self.lstm1(x_test)
                    # predict wide beam
                    y_test2, (hn2, cn2) = self.lstm2(x_test)

                # else, partial wide beam training
                else:
                    # select partial beams based on MPC
                    x_test = torch.zeros((batch_size, 2, 16)).to(device)
                    for b in range(batch_size):
                        x_test[b, :, max_id[b, :]] = x[b, :, i, max_id[b, :]]

                    #CNN
                    x_test = self.bn0(x_test)
                    x_test = self.conv1(x_test)
                    x_test = self.bn1(x_test)
                    x_test = F.relu(x_test)
                    x_test = self.conv2(x_test)
                    x_test = self.bn2(x_test)
                    x_test = F.relu(x_test)

                    P_dim_size = x_test.shape[2]
                    x_test = nn.MaxPool1d(kernel_size=P_dim_size)(x_test)

                    x_test = x_test.permute(2, 0, 1)
                    # predict narrow beam
                    y_test, (hn, cn) = self.lstm1(x_test, (hn, cn))
                    # predict wide beam
                    y_test2, (hn2, cn2) = self.lstm2(x_test, (hn2, cn2))

                # predict wide beam
                y_guide = self.drop2(y_test2)
                y_guide = self.fc2(y_guide)
                # predict narrow beam
                y_test = self.drop1(y_test)
                y_test = self.fc1(y_test)
                # MPC based beam selection
                max_value, max_id = torch.topk(y_guide, k)
                max_id = torch.squeeze(max_id)
                y1[i, :, :] = y_test
                y2[i, :, :] = y_guide

        # ONC
        # code structure is similar
        #if 0:
        k = 7
        batch_size = 16
        y1 = torch.zeros((10, 16, 64)).to(device)
        y2 = torch.zeros((10, 16, 16)).to(device)
        candidate_beam = torch.linspace(0, 15, steps=16)
        candidate_beam = candidate_beam.repeat(16, 1).to(device)

        for i in range(10):

            if i == 0:
                x_test = x[:, :, i, :]

                x_test = self.bn0(x_test)
                x_test = self.conv1(x_test)
                x_test = self.bn1(x_test)
                x_test = F.relu(x_test)
                x_test = self.conv2(x_test)
                x_test = self.bn2(x_test)
                x_test = F.relu(x_test)

                P_dim_size = x_test.shape[2]
                x_test = nn.MaxPool1d(kernel_size=P_dim_size)(x_test)

                x_test = x_test.permute(2, 0, 1)
                y_test, (hn, cn) = self.lstm1(x_test)
                y_test2, (hn2, cn2) = self.lstm2(x_test)

            else:
                x_test = torch.zeros((batch_size, 2, 16)).to(device)
                # ONC based beam selection
                for b in range(batch_size):
                    x_test[b, :, max_id[b, :]] = x[b, :, i, max_id[b, :]]

                x_test = self.bn0(x_test)
                x_test = self.conv1(x_test)
                x_test = self.bn1(x_test)
                x_test = F.relu(x_test)
                x_test = self.conv2(x_test)
                x_test = self.bn2(x_test)
                x_test = F.relu(x_test)

                P_dim_size = x_test.shape[2]
                x_test = nn.MaxPool1d(kernel_size=P_dim_size)(x_test)

                x_test = x_test.permute(2, 0, 1)
                y_test, (hn, cn) = self.lstm1(x_test, (hn, cn))
                y_test2, (hn2, cn2) = self.lstm2(x_test, (hn2, cn2))

            y_guide = self.drop2(y_test2)
            y_guide = self.fc2(y_guide)
            y_test = self.drop1(y_test)
            y_test = self.fc1(y_test)
            # ONC based beam selection
            max_value, max_id = torch.topk(y_guide, 1)
            max_id = torch.squeeze(max_id)
            max_id = max_id.repeat(16, 1).T
            max_value, max_id = torch.topk(-torch.abs(candidate_beam - max_id), k)
            y1[i, :, :] = y_test
            y2[i, :, :] = y_guide

        return y1, y2
