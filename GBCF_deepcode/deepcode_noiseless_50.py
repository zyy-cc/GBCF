import argparse
import math
import random
import torch
import torch.optim as optim
from random import randint
import numpy as np
import torch.nn.functional as F
from datetime import date

import matplotlib.pyplot as plt


identity = 'deepcode_noiseless_snr3'
print('[ID]', identity)

def get_args(jupyter_notebook):
    parser = argparse.ArgumentParser()
    parser.add_argument('-code_rate', type=int, default=3)
    parser.add_argument('-learning_rate', type = float, default=0.005)

    parser.add_argument('-batch_size', type=int, default=400)
    parser.add_argument('-num_epoch', type=int, default=200)

    parser.add_argument('-block_len', type=int, default=3)
    parser.add_argument('-num_samples_train', type=int, default=80000)
    parser.add_argument('-num_samples_validation', type=int, default=20000)

    parser.add_argument('-feedback_SNR1', type=int, default=100, help='100 means noiseless feeback')
    parser.add_argument('-feedback_SNR2', type=int, default=100, help='100 means noiseless feeback')
    parser.add_argument('-forward_SNR1', type=int, default=3) # forward SNR for user 1
    parser.add_argument('-forward_SNR2', type=int, default=3) # forward SNR for user 2

    parser.add_argument('-enc_num_unit',  type=int, default=50)
    parser.add_argument('-dec_num_unit',  type=int, default=50)
    parser.add_argument('-clip_norm', type = float, default=0.5)

    parser.add_argument('-enc_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid'], default='sigmoid')

    parser.add_argument('--zero_padding', action='store_true', default=False,
                        help='enable zero padding')

    parser.add_argument('--power_allocation', action='store_true', default=True,
                        help='enable power allocation')
    parser.add_argument('-with_cuda', type=bool, default=True)
    parser.add_argument('-initial_weights', type=str, default='default')

    args = parser.parse_args()

    if jupyter_notebook:
        args = parser.parse_args(args=[])  
    else:
        args = parser.parse_args()   

    return args

def snr_db_2_sigma(snr_db, feedback=False):
    if feedback and snr_db == 100:
        return 0
    return 10**(-snr_db * 1.0 / 20)


def errors_ber(y_true, y_pred, device, positions = 'default'):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    y_true = y_true.view(y_true.shape[0], -1, 1)    
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    comparisin_result = torch.ne(torch.round(y_true), torch.round(y_pred)).float()  
    res = torch.sum(comparisin_result)/(comparisin_result.shape[0]*comparisin_result.shape[1]*comparisin_result.shape[2]) 
    return res

def errors_bler(y_true, y_pred, device, positions = 'default'):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    y_true = y_true.view(y_true.shape[0], -1, 1)    # the size -1 is inferred from other dimensions
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)
    t1 = torch.round(y_true[:,:,:])
    t2 = torch.round(y_pred[:,:,:])

    decoded_bits = t1
    X_test       = t2
    tp0 = (abs(decoded_bits-X_test)).reshape([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    return bler_err_rate

def generate_data(args, train, forward_sigma, feedback_sigma):
    if train:
        num_samples = args.batch_size
    else:
        num_samples = args.num_samples_validation
    if args.zero_padding:
        X  = torch.randint(0, 2, (num_samples, args.block_len, 1))
        X = torch.cat([X, torch.zeros(num_samples, 1, 1)], dim=1)
        forward_noise = forward_sigma * torch.randn((num_samples, args.block_len+1, args.code_rate))
        feedback_noise= feedback_sigma * torch.randn((num_samples, args.block_len+1, args.code_rate))
    else:
        X  = torch.randint(0, 2, (num_samples, args.block_len, 1))
        forward_noise = forward_sigma * torch.randn((num_samples, args.block_len, args.code_rate))
        feedback_noise= feedback_sigma * torch.randn((num_samples, args.block_len, args.code_rate))

    return X, forward_noise, feedback_noise


def validation(model, device, X_1, forward_noise_1, feedback_noise_1, X_2, forward_noise_2, feedback_noise_2):

    model.eval()

    codewords, X1_pred, X2_pred = model(X_1, forward_noise_1, feedback_noise_1, X_2, forward_noise_2, feedback_noise_2)

    print('----------power--------')
    print('codewords with mean:  ', torch.mean(codewords).cpu().detach().numpy())
    print('codewords with power: ', torch.var(codewords).cpu().detach().numpy())

    codewords_stat = codewords[:,:,0].cpu().detach().numpy()
    print('first codewords with mean:  ', np.mean(codewords_stat))
    print('first codewords with power: ', np.var(codewords_stat))

    codewords_stat = codewords[:,:,1].cpu().detach().numpy()
    print('second codewords with mean:  ', np.mean(codewords_stat))
    print('second codewords with power: ', np.var(codewords_stat))

    codewords_stat = codewords[:,:,2].cpu().detach().numpy()
    print('third codewords with mean:  ', np.mean(codewords_stat))
    print('third codewords with power: ', np.var(codewords_stat))

    X1_pred = torch.clamp(X1_pred, 0.0, 1.0)
    X2_pred = torch.clamp(X2_pred, 0.0, 1.0)
    X_1 = X_1.float()
    X_2 = X_2.float()
    
    loss1 = torch.nn.functional.binary_cross_entropy(X1_pred , X_1)
    loss2 = torch.nn.functional.binary_cross_entropy(X2_pred , X_2)

    loss = loss1 + loss2 

    X1_pred = X1_pred.cpu().detach()
    X2_pred = X2_pred.cpu().detach()
    

    ber_1 = errors_ber(X_1, X1_pred, device)
    ber_2 = errors_ber(X_2, X2_pred, device)

    bler1 = errors_bler(X_1, X1_pred, device)
    bler2 = errors_bler(X_2, X2_pred, device)

    return loss.item(), ber_1.item(), ber_2.item(), bler1, bler2, codewords

def train(args, model, device, optimizer, scheduler):  
    model.train()

    loss_train = 0.0
    train_loss1 = 0.0
    train_loss2 = 0.0

    num_batch = int(args.num_samples_train/args.batch_size)

    for __ in range(num_batch):
        X1, forward_noise1, feedback_noise1 = generate_data(args, True, snr_db_2_sigma(args.forward_SNR1), snr_db_2_sigma(args.feedback_SNR1, True))
        X2, forward_noise2, feedback_noise2 = generate_data(args, True, snr_db_2_sigma(args.forward_SNR2), snr_db_2_sigma(args.feedback_SNR2, True))
        X1, forward_noise1, feedback_noise1, X2, forward_noise2, feedback_noise2 = X1.to(device), forward_noise1.to(device), feedback_noise1.to(device),  X2.to(device), forward_noise2.to(device), feedback_noise2.to(device)
      
        optimizer.zero_grad()

        codewords, X1_pred, X2_pred = model(X1, forward_noise1, feedback_noise1, X2, forward_noise2, feedback_noise2)

        X1 = X1.float()
        X2 = X2.float()
    
        loss1 = F.binary_cross_entropy(X1_pred, X1)
        
        train_loss1 += loss1.item()
        loss2 = F.binary_cross_entropy(X2_pred, X2)
        train_loss2 += loss2.item()

        loss = loss1 + loss2

        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        optimizer.step()
        loss_train = loss_train + loss.item()
    loss1 = train_loss1/num_batch
    loss2 = train_loss2/num_batch
    loss_train = loss_train / num_batch
    return loss1, loss2, loss_train



class Power_reallocate(torch.nn.Module): 
    def __init__(self, args):
        super(Power_reallocate, self).__init__()
        self.args = args

        req_grad = True if args.power_allocation else False
        if args.zero_padding:
            self.weight = torch.nn.Parameter(torch.Tensor(args.block_len+1, args.code_rate),requires_grad = req_grad)
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(args.block_len, args.code_rate),requires_grad = req_grad )
        self.weight.data.uniform_(1.0, 1.0)
       
    def forward(self, inputs, phase = -1,idx = -1):
        if args.zero_padding:
            self.wt   = torch.sqrt(self.weight**2 * ((args.block_len+1) * args.code_rate) / torch.sum(self.weight**2))
        else:
            self.wt   = torch.sqrt(self.weight**2 * (args.block_len * args.code_rate) / torch.sum(self.weight**2))


        # element-wise multiple the weight and inputs 
        if phase == 0 or phase == 1: 
            res = torch.mul(inputs, self.wt[:, phase].view(1, -1, 1))
        else:
            res = torch.mul(inputs, self.wt[idx, 2].view(1, 1, 1))
        return res


class AE(torch.nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()

        self.args             = args

        # Encoder
        self.enc_gru1   = torch.nn.GRU(input_size=8, hidden_size=args.enc_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.enc_linear    = torch.nn.Linear(in_features=args.enc_num_unit, out_features=1, bias=True) 

        # Decoder
        # user 1
        self.dec_gru_1          = torch.nn.GRU(input_size = args.code_rate,  hidden_size = args.dec_num_unit,num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.dec_linear1        = torch.nn.Linear(in_features=2*args.dec_num_unit, out_features=1, bias=True)

        # user 2
        self.dec_gru_2          = torch.nn.GRU(input_size = args.code_rate,  hidden_size = args.dec_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.dec_linear2        = torch.nn.Linear(in_features=2*args.dec_num_unit, out_features=1, bias=True)

        # power allocation
        self.total_power_reloc = Power_reallocate(args)


    def normalize(self, data):
       
        batch_mean = torch.mean(data, 0)
        batch_std = torch.std(data,0)
        data_normalized = (data - batch_mean)*1.0/batch_std

        return data_normalized

    def forward(self, X1, forward_noise1, feedback_noise1,  X2, forward_noise2, feedback_noise2):
        num_samples_input = X1.shape[0]
        block_len = X1.shape[1]

        # encoder part: Phase 1
        codewords_1      = 2.0 * X1 - 1.0 
        codewords_2      = 2.0 * X2 - 1.0

        norm_codewords_1 = self.normalize(codewords_1)
        norm_codewords_2 = self.normalize(codewords_2)

        power_codewords_1  = self.total_power_reloc(norm_codewords_1, 0)
        power_codewords_2  = self.total_power_reloc(norm_codewords_2, 1)

        
        phase1_feedback1 = power_codewords_1 + forward_noise1[:,:,0].view(num_samples_input, block_len, 1) + feedback_noise1[:,:,0].view(num_samples_input, block_len, 1)
        phase1_feedback2 = power_codewords_1 + forward_noise2[:,:,0].view(num_samples_input, block_len, 1) + feedback_noise2[:,:,0].view(num_samples_input, block_len, 1)

        phase2_feedback1 = power_codewords_2 + forward_noise1[:,:,1].view(num_samples_input, block_len, 1) + feedback_noise1[:,:,1].view(num_samples_input, block_len, 1)
        phase2_feedback2 = power_codewords_2 + forward_noise2[:,:,1].view(num_samples_input, block_len, 1) + feedback_noise2[:,:,1].view(num_samples_input, block_len, 1)

        # encoder part: Phase 2 generate parity bits
    
        for idx in range(block_len):
            if idx == 0:
                input_tmp        = torch.cat([X1[:,idx,:].view(num_samples_input, 1, 1),
                                              X2[:,idx,:].view(num_samples_input, 1, 1),
                                              phase1_feedback1[:,idx,:].view(num_samples_input, 1, 1),
                                              phase1_feedback2[:,idx,:].view(num_samples_input, 1, 1),
                                              phase2_feedback1[:,idx,:].view(num_samples_input, 1, 1),
                                              phase2_feedback2[:,idx,:].view(num_samples_input, 1, 1),
                                              torch.zeros((num_samples_input, 1, 2)).to(device)], dim=2)
                rnn_output1, h_tmp  = self.enc_gru1(input_tmp)
                dense_output         = torch.sigmoid(self.enc_linear(rnn_output1))

            else:
                input_tmp        = torch.cat([X1[:,idx,:].view(num_samples_input, 1, 1),
                                              X2[:,idx,:].view(num_samples_input, 1, 1),
                                              phase1_feedback1[:,idx,:].view(num_samples_input, 1, 1),
                                              phase1_feedback2[:,idx,:].view(num_samples_input, 1, 1),
                                              phase2_feedback1[:,idx,:].view(num_samples_input, 1, 1),
                                              phase2_feedback2[:,idx,:].view(num_samples_input, 1, 1),
                                              phase3_feedback1.view(num_samples_input, 1, 1),
                                              phase3_feedback2.view(num_samples_input, 1, 1)], dim=2)

                rnn_output1, h_tmp  = self.enc_gru1(input_tmp, h_tmp)  
                dense_output         = torch.sigmoid(self.enc_linear(rnn_output1))
            norm_phase3_output  = self.normalize(dense_output)
            power_phase3_output  = self.total_power_reloc(norm_phase3_output, 2, idx)
            
            phase3_feedback1 = power_phase3_output + forward_noise1[:,idx, 2].view(num_samples_input, 1, 1) + feedback_noise1[:,idx, 2].view(num_samples_input, 1, 1)
            phase3_feedback2 = power_phase3_output + forward_noise2[:,idx, 2].view(num_samples_input, 1, 1) + feedback_noise2[:,idx, 2].view(num_samples_input, 1, 1)

            if idx == 0:
                codewords_phase_3 = power_phase3_output.view(num_samples_input, 1, 1)
            else:
                codewords_phase_3 = torch.cat([codewords_phase_3, power_phase3_output], dim = 1)

        codewords =  torch.cat([power_codewords_1, power_codewords_2, codewords_phase_3], axis=2)

        # AWGN channel
        noisy_codewords1 =  codewords + forward_noise1
        noisy_codewords2 =  codewords + forward_noise2

        # decoder
        ####user 1
        gru_output_1, _  = self.dec_gru_1(noisy_codewords1)
        decoder_output1     = torch.sigmoid(self.dec_linear1(gru_output_1))

        ####user 2
        gru_output_2, _  = self.dec_gru_2(noisy_codewords2)
        decoder_output2     = torch.sigmoid(self.dec_linear2(gru_output_2))

        return codewords, decoder_output1, decoder_output2 
        
args = get_args(jupyter_notebook = False)
print('args = ', args.__dict__)

use_cuda = args.with_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('use_cuda = ', use_cuda)
print('device = ', device)

if use_cuda:
    model = AE(args).to(device)
else:
    model = AE(args)
print(model)

args.initial_weights = 'weights/deepcode_bc_noiseless_snr3.pt'
if args.initial_weights == 'default':
    pass
elif args.initial_weights == 'deepcode':
    f_load_deepcode_weights(model)
    print('deepcode weights are loaded.')
else:
    model.load_state_dict(torch.load(args.initial_weights, map_location=torch.device('cpu')))
    model.args = args
    print('initial weights are loaded.')

weight_decay=1e-11
optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate, betas=(0.9,0.999), eps=1e-07, weight_decay=0, amsgrad=False)
learning_rate_step_size = int(10**3 / args.batch_size)
print('learning_rate_step_size = ', learning_rate_step_size)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step_size, gamma=0.1)

X1, forward_noise1, feedback_noise1 = generate_data(args, False, snr_db_2_sigma(args.forward_SNR1), snr_db_2_sigma(args.feedback_SNR1, True))
X2, forward_noise2, feedback_noise2 = generate_data(args, False, snr_db_2_sigma(args.forward_SNR2), snr_db_2_sigma(args.feedback_SNR2, True))
X1, forward_noise1, feedback_noise1, X2, forward_noise2, feedback_noise2 = X1.to(device), forward_noise1.to(device), feedback_noise1.to(device),  X2.to(device), forward_noise2.to(device), feedback_noise2.to(device)


loss_his, ber1, ber2,  bler1, bler2, codewords= validation(model, device, X1, forward_noise1, feedback_noise1, X2, forward_noise2, feedback_noise2)

print('----- Ber 1(initial): ', ber1)
print('----- Ber 2(initial): ', ber2)
print('----- Bler 1(initial): ', bler1)
print('----- Bler 2(initial): ', bler2)
print('----- Validation loss (initial): ', loss_his)


# nums = 100
# ber1_list = []
# ber2_list = []
# bler1_list = []
# bler2_list = []

# for i in range(nums):
# 	print(f"-----------------------{i}------------------------")
# 	X1, forward_noise1, feedback_noise1 = generate_data(args, False, snr_db_2_sigma(args.forward_SNR1), snr_db_2_sigma(args.feedback_SNR1, True))
# 	X2, forward_noise2, feedback_noise2 = generate_data(args, False, snr_db_2_sigma(args.forward_SNR1), snr_db_2_sigma(args.feedback_SNR1, True))
# 	X1, forward_noise1, feedback_noise1, X2, forward_noise2, feedback_noise2 = X1.to(device), forward_noise1.to(device), feedback_noise1.to(device),  X2.to(device), forward_noise2.to(device), feedback_noise2.to(device)

# 	loss_his, ber_1, ber_2,  bler1, bler2, codewords= validation(model, device, X1, forward_noise1, feedback_noise1, X2, forward_noise2, feedback_noise2)


# 	print('----- Ber 1(initial): ', ber_1)
# 	print('----- Ber 2(initial): ', ber_2)

# 	print('the BLER for user 1 is:', bler1)
# 	print('the BLER for user 2 is:', bler2)


# 	print("total power", torch.var(codewords))
# 	ber1_list.append(ber_1)
# 	ber2_list.append(ber_2)
# 	bler1_list.append(bler1)
# 	bler2_list.append(bler2)

# avg_ber1 = sum(ber1_list)/nums
# avg_ber2 = sum(ber2_list)/nums

# avg_bler1 = sum(bler1_list)/nums
# avg_bler2 = sum(bler2_list)/nums
# print('----- avg_ber1: ', avg_ber1)
# print('----- avg_ber2: ', avg_ber2)
# print("-----avg ber", (avg_ber1 + avg_ber2)/2)

# print('----- avg_bler1: ', avg_bler1)
# print('----- avg_bler2: ', avg_bler2)
# print("-----avg bler",  (avg_bler1 + avg_bler2)/2)




# for epoch in range(1, args.num_epoch + 1):
#     train_loss1, train_loss2, total_loss = train(args, model, device, optimizer, scheduler)
#     print('====> Epoch: {} Average loss: {:.4f} for user 1, average loss: {:.4f} for user 2, total loss:{:.4f}'.format(epoch, train_loss1, train_loss2, total_loss) )
#     if epoch%10 == 0:
#         loss_his, ber_1, ber_2,  bler1, bler2, codewords = validation(model, device, X1, forward_noise1, feedback_noise1, X2, forward_noise2, feedback_noise2)
#         print('----- Ber 1(initial): ', ber_1)
#         print('----- Ber 2(initial): ', ber_2)
#         print('----- Validation loss (initial): ', loss_his)
        
#         file_name = './logs/model_'+date.today().strftime("%Y%m%d")+'_'+identity+'.pt'
#         torch.save(model.state_dict(), file_name)
#         print('saved model as file: ', file_name)

# print('final saved file_name = ', file_name)
