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
    BL = np.zeros((10,5))
    running_loss = 0
    batch_num = 0
    rank = np.zeros((10,64))
    pdf = np.zeros((10, 101))
    batch_size = 16
    # save original predicted probabilities
    p = np.zeros((4096, 16))
    while not done:
        channel1, channel2, label_m, beam_power_m, beam_power_nonoise_m, label_nonoise_m, label_widebeam_m, done, count = loader.next_batch()
        if count==True:
            batch_num += 1
            out_tensor, out_tensor2  = model(channel2, device)
            loss = 0
            for loss_count in range(10):
                loss += criterion(torch.squeeze(out_tensor[loss_count, :, :]), label_nonoise_m[:, loss_count])
            out_tensor_np = out_tensor.cpu().detach().numpy()
            out_tensor2_np = out_tensor2.cpu().detach().numpy()
            gt_labels = label_nonoise_m.cpu().detach().numpy()
            gt_labels = np.float32(gt_labels)
            gt_labels = gt_labels.transpose(1, 0)
            beam_power = beam_power_nonoise_m.cpu().detach().numpy()
            beam_power = beam_power.transpose(1, 0, 2)
            out_shape = gt_labels.shape

            # save predicted probabilities
            p[(batch_num - 1) * batch_size: batch_num * batch_size, :] = np.squeeze(out_tensor2_np[0, :, :])

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
                    BL[i, 1] = BL[i, 1] + (max(beam_power[i, j, train_sorted[62 : 64]]) / max(beam_power[i, j, :])) ** 2
                    BL[i, 2] = BL[i, 2] + (max(beam_power[i, j, train_sorted[61 : 64]]) / max(beam_power[i, j, :])) ** 2
                    BL[i, 3] = BL[i, 3] + (max(beam_power[i, j, train_sorted[60 : 64]]) / max(beam_power[i, j, :])) ** 2
                    BL[i, 4] = BL[i, 4] + (max(beam_power[i, j, train_sorted[59 : 64]]) / max(beam_power[i, j, :])) ** 2
                    pdf_index = np.floor(((beam_power[i, j, train_index] / max(beam_power[i, j, :])) ** 2) * 100)
                    pdf_index = pdf_index.astype(int)
                    pdf[i, pdf_index : 101] = pdf[i, pdf_index : 101] + 1

            running_loss += loss.data.cpu()
    acur = float(P) / (P + N)
    losses = running_loss / batch_num
    BL = BL / batch_num / 16
    print("Accuracy: %.3f" % (acur))
    print("Loss: %.3f" % (losses))
    print("Beam power loss:")
    print(BL.T)
    print(pdf)
    return acur, losses, rank, BL, pdf, p

def main():
    version_name = 'v1_K=3_probability'
    info = 'TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_MPC_' + version_name
    print(info)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    t = 1
    batch_size = 16
    
    print('batch_size:%d'%(batch_size))

    acur_eval = []
    loss_eval = []
    BL_eval = np.zeros((10, 5, t))
    rank_eval = np.zeros((10, 64, t))
    pdf_eval = np.zeros((10, 101, t))
    pp = np.zeros((4096, 16, t))


    for tt in range(t):
        print('Train %d times' % (tt))
        model_name = 'TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_MPC_v1_k3_evaluation_MODEL.pkl'
        model = torch.load(model_name)
        model.to(device)
        model.eval()

        eval_loader = Dataloader(path='/usr/mk/TCOM/dataset/testing_15dBm', batch_size=batch_size, device=device)
        eval_loader.reset()
        acur, losses, rank, BL, pdf, p = eval(model, eval_loader, device)
        acur_eval.append(acur)
        loss_eval.append(losses)
        rank_eval[:, :, tt] = np.squeeze(rank)
        BL_eval[:, :, tt] = np.squeeze(BL)
        pdf_eval[:, :, tt] = np.squeeze(pdf)
        pp[:, :, tt] = np.squeeze(p)

        mat_name = info + '.mat'
        sio.savemat(mat_name, {'acur_eval': acur_eval, 'loss_eval': loss_eval,
                               'rank_eval': rank_eval, 'BL_eval': BL_eval, 'pdf_eval': pdf_eval, 'p_eval': pp})
    
if __name__ == '__main__':
    main()