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
    loader.reset()
    criterion = nn.CrossEntropyLoss()
    done = False
    P = 0
    N = 0
    #M = 0
    batch_size = 16
    BL = np.zeros((10,1))
    running_loss = 0
    batch_num = 0
    rank = np.zeros((10,64))
    while not done:
        channel1, channel2, label_m, beam_power_m, beam_power_nonoise_m, label_nonoise_m, label_widebeam_m, done, count = loader.next_batch()
        if count==True:
            batch_num += 1
            out_tensor, out_tensor2 = model(channel2, device)
            loss = 0
            for loss_count in range(10):
                loss += criterion(torch.squeeze(out_tensor[loss_count, :, :]), label_nonoise_m[:, loss_count])
            out_tensor_np = out_tensor.cpu().detach().numpy()
            gt_labels = label_nonoise_m.cpu().detach().numpy()
            gt_labels = np.float32(gt_labels)
            gt_labels = gt_labels.transpose(1, 0)
            beam_power = beam_power_nonoise_m.cpu().detach().numpy()
            beam_power = beam_power.transpose(1, 0, 2)
            out_shape = gt_labels.shape
            for i in range(out_shape[0]):
                for j in range(out_shape[1]):
                    train_ans = np.squeeze(out_tensor_np[i, j, :])
                    train_index = np.argmax(train_ans)
                    train_sorted = np.argsort(train_ans)
                    rank_index = np.where(train_sorted == gt_labels[i, j])
                    rank[i, rank_index[0]] = rank[i, rank_index[0]] + 1
                    if train_index == gt_labels[i, j]:
                        P = P + 1
                    else:
                        N = N + 1
                    BL[i, 0] = BL[i, 0] + (beam_power[i, j, train_index] / max(beam_power[i, j, :])) ** 2
            running_loss += loss.data.cpu()
    acur = float(P) / (P + N)
    losses = running_loss / batch_num
    BL = BL / batch_num / batch_size
    print("Accuracy: %.3f" % (acur))
    print("Loss: %.3f" % (losses))
    print("Beam power loss:")
    print(BL.T)
    return acur, losses, rank, BL

def main():
    version_name = 'v1_k=7_velocity'
    info = 'TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_' + version_name
    print(info)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    t = 5
    batch_size = 16
    
    print('batch_size:%d'%(batch_size))

    acur_eval = []
    loss_eval = []
    BL_eval = np.zeros((10, 5, t))
    rank_eval = np.zeros((10, 64, 5, t))

    for tt in range(t):
        print('Train %d times' % (tt))
        model_name = 'TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_k=7_' + str(tt) + '_MODEL.pkl'
        model = torch.load(model_name)
        model.to(device)
        model.eval()
        count = 0

        # evaluate the performance under different UE velocities
        for v in range(10, 60, 10):
            eval_loader_name = '/usr/mk/TCOM/dataset/velocity_' + str(v)
            eval_loader = Dataloader(path=eval_loader_name, batch_size=batch_size, device=device)
            eval_loader.reset()
            acur, losses, rank, BL = eval(model, eval_loader, device)
            acur_eval.append(acur)
            loss_eval.append(losses)
            rank_eval[:, :, count, tt] = np.squeeze(rank)
            BL_eval[:, count, tt] = np.squeeze(BL)
            count = count + 1

            mat_name = info + '.mat'
            sio.savemat(mat_name, {'acur_eval': acur_eval, 'loss_eval': loss_eval, 'rank_eval': rank_eval, 'BL_eval': BL_eval})
    
if __name__ == '__main__':
    main()