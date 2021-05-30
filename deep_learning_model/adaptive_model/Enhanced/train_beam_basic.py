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

# detailed code comments can be seen in the folder basic_model
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
            out_tensor1, out_tensor2 = model(channel2, device)
            loss = 0
            # for performance evaluation, only the loss of narrow beam predictions is focused
            for loss_count in range(10):
                loss += criterion(torch.squeeze(out_tensor1[loss_count, :, :]), label_nonoise_m[:, loss_count])
            out_tensor_np = out_tensor1.cpu().detach().numpy()
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
    version_name = 'v1_k=7'
    info = 'TCOM_LOS_64beam_2CNN_1LSTM_256feature_16Tx_RK=8dB_proposed3_ONC_' + version_name
    print(info)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    t = 5
    epoch = 80
    batch_size = 16
    
    print('batch_size:%d'%(batch_size))
    loader = Dataloader(path='/usr/mk/TCOM/dataset/training_15dBm', batch_size=batch_size, device=device)
    eval_loader = Dataloader(path='/usr/mk/TCOM/dataset/testing_15dBm', batch_size=batch_size, device=device)

    criterion = nn.CrossEntropyLoss()

    for tt in range(t):
        print('Train %d times' % (tt))
        lr = 0.0001 # learning rate
        model = Model_3D(N=15, K=32, Tx=4, Channel=2)
        model.to(device)
        min_loss = 1000000
        optimizer = torch.optim.Adam(model.parameters(), lr / 10, betas=(0.9, 0.999)) # use the sum of 10 losses
        lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                              verbose=True, threshold=0.0001,
                                                              threshold_mode='rel', cooldown=0, min_lr=0.0000001,
                                                              eps=1e-08)
        for name, param in model.named_parameters():
            print('Name:', name, 'Size:', param.size())

        for e in range(epoch):
            print('Train %d epoch'%(e))
            loader.reset()
            eval_loader.reset()
            done = False
            running_loss = 0
            batch_num = 0
            while not done:
                channel1, channel2, labels, beam_power_m, beam_power_nonoise_m, label_nonoise_m, labels_widebeam_m, done, count = loader.next_batch()
                if count == True:
                    batch_num += 1
                    out_tensor1, out_tensor2 = model(channel2, device)
                    loss = 0
                    # for model training, the total loss is the sum of narrow and wide beam predictions
                    for loss_count in range(10):
                        loss += criterion(torch.squeeze(out_tensor1[loss_count, :, :]), labels[:, loss_count])
                    for loss_count in range(9):
                        loss += criterion(torch.squeeze(out_tensor2[loss_count, :, :]), labels_widebeam_m[:, loss_count + 1])
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            losses = running_loss / batch_num
            print('[%d] loss: %.3f' %
                      (e + 1, losses))
            model.eval()
            print('the evaling set:')
            acur, losses, rank, BL = eval(model, eval_loader, device)
            if losses < min_loss:
                min_loss = losses
                model_name = info + '_' + str(tt) + '_MODEL.pkl'
                torch.save(model, model_name)
            lr_decay.step(losses)
            model.train()
    
if __name__ == '__main__':
    main()