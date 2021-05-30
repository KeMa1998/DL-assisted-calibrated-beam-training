import torch.optim as optim
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from dataloader import Dataloader
from model_3Dcov_basic import Model_3D
import sys

def eval(model, loader, device):
    # reset dataloader
    loader.reset()
    # loss function
    criterion = nn.CrossEntropyLoss()
    # judge whether dataset is finished
    done = False
    # counting accurate prediction
    P = 0
    # counting inaccurate prediction
    N = 0
    # M = 0
    # beam power loss
    BL = np.zeros((10,1))
    # running loss
    running_loss = 0
    # count batch number
    batch_num = 0
    batch_size = 16
    # count the rank of the predicted probabilities
    rank = np.zeros((10,64))
    # evaluate validation set
    while not done:
        # read files
        channel1, channel2, label_m, beam_power_m, beam_power_nonoise_m, label_nonoise_m, done, count = loader.next_batch()
        if count==True:
            batch_num += 1
            # channel1: baseline 1 by using uniformly sampled beams
            # channel2: proposed by using wide beams
            out_tensor = model(channel2)
            loss = 0
            # average loss of all predictions
            for loss_count in range(10):
                loss += criterion(torch.squeeze(out_tensor[loss_count, :, :]), label_nonoise_m[:, loss_count])
            # predicted probabilities
            out_tensor_np = out_tensor.cpu().detach().numpy()
            # true label without noise
            gt_labels = label_nonoise_m.cpu().detach().numpy()
            gt_labels = np.float32(gt_labels)
            gt_labels = gt_labels.transpose(1, 0)
            # true normalized beam amplitude
            beam_power = beam_power_nonoise_m.cpu().detach().numpy()
            beam_power = beam_power.transpose(1, 0, 2)
            out_shape = gt_labels.shape
            for i in range(out_shape[0]):
                for j in range(out_shape[1]):
                    train_ans = np.squeeze(out_tensor_np[i, j, :])
                    # find the index with the maximum probability
                    train_index = np.argmax(train_ans)
                    # find the rank of the true optimal beam
                    train_sorted = np.argsort(train_ans)
                    rank_index = np.where(train_sorted == gt_labels[i, j])
                    rank[i, rank_index[0]] = rank[i, rank_index[0]] + 1
                    # counting accurate and inaccurate prediction
                    if train_index == gt_labels[i, j]:
                        P = P + 1
                    else:
                        N = N + 1
                    # calculate beam power loss
                    BL[i, 0] = BL[i, 0] + (beam_power[i, j, train_index] / max(beam_power[i, j, :])) ** 2
            running_loss += loss.data.cpu()
    # average accuracy
    acur = float(P) / (P + N)
    # average loss
    losses = running_loss / batch_num
    # average loss
    BL = BL / batch_num / batch_size
    # print results
    print("Accuracy: %.3f" % (acur))
    print("Loss: %.3f" % (losses))
    print("Beam power loss:")
    print(BL.T)
    return acur, losses, rank, BL

def main():
    # 0 LSTM: baseline 2 or proposed CNN based
    # 1 LSTM: proposed LSTM based
    version_name = 'v1'
    info = 'TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed2_' + version_name
    print(info)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # training time
    t = 5
    # training epoch
    epoch = 80
    # batch size
    batch_size = 16
    print('batch_size:%d'%(batch_size))

    # training set and validation set
    loader = Dataloader(path='/usr/mk/TCOM/dataset/training_15dBm', batch_size=batch_size, device=device)
    eval_loader = Dataloader(path='/usr/mk/TCOM/dataset/testing_15dBm', batch_size=batch_size, device=device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # first loop for training times
    for tt in range(t):
        print('Train %d times' % (tt))
        # learning rate
        lr = 0.0003
        # model initialization
        model = Model_3D(N=15, K=32, Tx=4, Channel=2)
        model.to(device)
        # save minimum loss
        min_loss = 1000000
        # Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr / 10, betas=(0.9, 0.999)) # use the sum of 10 losses
        # learning rate adaptive decay
        lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                              verbose=True, threshold=0.0001,
                                                              threshold_mode='rel', cooldown=0, min_lr=0.0000001,
                                                              eps=1e-08)
        # print parameters
        for name, param in model.named_parameters():
            print('Name:', name, 'Size:', param.size())

        # second loop for training times
        for e in range(epoch):
            print('Train %d epoch'%(e))
            # reset the dataloader
            loader.reset()
            eval_loader.reset()
            # judge whether data loading is done
            done = False
            # running loss
            running_loss = 0
            # count batch number
            batch_num = 0
            while not done:
                # read files
                channel1, channel2, labels, beam_power_m, beam_power_nonoise_m, label_nonoise_m, done, count = loader.next_batch()
                if count == True:
                    batch_num += 1
                    # predicted probabilities
                    out_tensor = model(channel2)
                    loss = 0
                    # average loss of all predictions
                    for loss_count in range(10):
                        loss += criterion(torch.squeeze(out_tensor[loss_count, :, :]), labels[:, loss_count])
                    # gradient back propagation
                    loss.backward()
                    # parameter optimization
                    optimizer.step()
                    running_loss += loss.item()
            losses = running_loss / batch_num
            #print results
            print('[%d] loss: %.3f' %
                      (e + 1, losses))
            # eval mode, where dropout is off
            model.eval()
            print('the evaling set:')
            acur, losses, rank, BL = eval(model, eval_loader, device)
            # save the optimal model
            if losses < min_loss:
                min_loss = losses
                model_name = info + '_' + str(tt) + '_MODEL.pkl'
                torch.save(model, model_name)
            # learning rate decay
            lr_decay.step(losses)
            # train mode, where dropout is on
            model.train()
    
if __name__ == '__main__':
    main()