"""
first two transmission: PAM modulation
[codewords, noisy codewords1, noisy codewords2]
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import date
import os
import math, copy


torch.set_default_dtype(torch.float32)

# identity = 'lightcode_bc_snr3'
# print('[ID]', identity)


def get_args():
    parser = argparse.ArgumentParser()

    # channel configuration
    parser.add_argument('-forward_SNR1', type=int, default=2)
    parser.add_argument('-forward_SNR2', type=int, default=2)
    parser.add_argument('-feedback_SNR1', type=int, default=100, help='100 means noiseless feeback')  
    parser.add_argument('-feedback_SNR2', type=int, default=100, help='100 means noiseless feeback')  

    parser.add_argument('-K1', type=int, default=3)
    parser.add_argument('-K2', type=int, default=3)
    parser.add_argument('-N', type=int, default=9, help='Number of transmission rounds')

    parser.add_argument('-enc_input_size', type=int, default=24) # 3(N-1)
    parser.add_argument('-enc_hidden_size', type=int, default=32)
    parser.add_argument('-dec_hidden_size', type=int, default=32)
    parser.add_argument('-power_allocation', type=bool, default=True)
    parser.add_argument('-num_samples_test', type=int, default=100000)  # 2000000


    # training configuration
    parser.add_argument('-totalbatch', type=int, default=4000, help="number of total batches to train") # or larger
    parser.add_argument('-num_epoch', type=int, default=1, help="number of epochs to train")
    parser.add_argument('-batch_size', type=int, default=100000, help="batch size")
    parser.add_argument('-lr', type=float, default=0.005, help="learning rate") # 0.02
    parser.add_argument('-clip_th', type=float, default=0.5, help="clipping threshold")
    parser.add_argument('-initial_weights', type=str, default='default')


    # Learning arguments
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay")
    parser.add_argument('-with_cuda', type=bool, default=True)
    args = parser.parse_args()

    return args

def snr_db_2_sigma(snr_db, feedback=False):
    if feedback and snr_db == 100:
        return 0
    return 10**(-snr_db * 1.0 / 20)

def generate_data(num_samples, K, N, forward_SNR, feedback_SNR):
    """
    K: length of message bits
    N: number of rounds
    Output: 
    message (num_samples, K)
    forward_noise (num_samples, N)
    feedback_noise (num_samples, N)
    """
    forward_sigma = snr_db_2_sigma(forward_SNR)
    feedback_sigma = snr_db_2_sigma(feedback_SNR, feedback=True)
    message = torch.randint(0, 2, (num_samples, K))
    forward_noise = forward_sigma * torch.randn((num_samples, N))
    feedback_noise = feedback_sigma * torch.randn((num_samples, N))
    return message, forward_noise, feedback_noise

def bin2dec(binary_data, k):
    """
    Transform the binary message bits to real value.
    Input: (num_samples, k)
    Output: (num_samples, 1)
    """
    power = (2 ** torch.arange(k - 1, -1, -1, device=binary_data.device, dtype=binary_data.dtype)).float()
    decimal_output = torch.matmul(binary_data.float(), power).unsqueeze(-1)
    
    return decimal_output

def dec2bin(decimal_data, k):
    """
    transform the real value to message bits
    Input: (num_samples, 1)
    Output: (num_samples, k)
    """
    power = 2 ** torch.arange(k - 1, -1, -1).to(decimal_data.device)
    boolean = torch.bitwise_and(decimal_data, power) > 0
    binary_output = boolean.to(dtype=torch.int64)
    return binary_output



def PAMmodulation(binary_data, k):
    """
    Input: m (num_samples, k)
    Output: theta (num_samples, 1)
    """

    M = 2**k
    decimal_data = bin2dec(binary_data, k)
    eta = torch.sqrt(torch.tensor(3)/(M**2 -1))
    theta = (2 * decimal_data - (M - 1)) * eta 

    return decimal_data, theta


def PAMdedulation(noisy_theta, k):
    """
    Input: noisy theta (num_samples, 1)
    Output: message bits (num_samples, k)
    """
    M = 2**k
    eta = torch.sqrt(torch.tensor(3)/(M**2 -1))
    noisy_theta_clamp = torch.clamp(noisy_theta, min = -(M-1)*eta, max = (M-1)*eta)
    decimal_data = torch.round((noisy_theta_clamp/eta + M-1)/2).to(dtype=torch.int64)
    decoding_output = dec2bin(decimal_data, k)
    return decimal_data, decoding_output

def normalize(theta, P):
	"""
	normalize data to satisfy power constraint P
	Input: theta (num_samples, 1)
	Output: theta (num_samples, 1)
	"""
	# normalize the data based on data
	theta_mean = torch.mean(theta, 0)
	theta_std = torch.std(theta,0)
	normalized_theta = torch.sqrt(P)  * ((theta - theta_mean)*1.0/theta_std) 
	return normalized_theta


class FE(nn.Module):
    def __init__(self, args, mod, input_size, d_model):
        super(FE, self).__init__()
        self.args = args
        self.mod = mod

        if mod == 'enc':
            self.activation = nn.ReLU()
            self.u1_FC1 = nn.Linear(input_size, 2 * d_model, bias=True)
            self.u1_FC2 = nn.Linear(2 * d_model, 2 * d_model, bias=True)
            self.u1_FC3 = nn.Linear(2 * d_model, 2 * d_model, bias=True)
            self.u1_FC4 = nn.Linear(4 * d_model, d_model, bias = True)
            

        elif mod == 'dec':
            self.activation = nn.ReLU()
            self.FC1 = nn.Linear(input_size, 2 * d_model, bias=True)
            self.FC2 = nn.Linear(2 * d_model, 2 * d_model, bias=True)
            self.FC3 = nn.Linear(2 * d_model, 2 * d_model, bias=True)
            self.FC4 = nn.Linear(4 * d_model, d_model, bias = True)
        else:
            raise ValueError('Invalid mode')

    def forward(self, src):
        """
        (batch_size, input_size) -> (batch_size, hidden_size)
        """

        if self.mod == 'enc':
        
            user1_x1 = self.u1_FC1(src)
            user1_x2 = self.u1_FC2(self.activation(user1_x1))
            user1_x3 = self.u1_FC3(self.activation(user1_x2))
            user1_x4 = self.u1_FC4(torch.cat([user1_x3, -1 * user1_x1], dim = 1))

            return user1_x4

        elif self.mod == 'dec':
            x1 = self.FC1(src)
            x2 = self.FC2(self.activation(x1))
            x3 = self.FC3(self.activation(x2))
            x = self.FC4(torch.cat([x3, -1 * x1], dim = 1))
        else:
            raise ValueError('Invalid mode')
        return x
        

class MLP(nn.Module):
    def __init__(self, args, mod, input_size, d_model, user=-1):
        super(MLP, self).__init__()
        self.args = args
        self.mod = mod
        self.fe = FE(args, mod, input_size, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.user = user
        if mod == 'enc':
            # encoder output
            d_model_reduced = int(d_model/4)
            self.enc_out1 = nn.Linear(d_model, d_model_reduced, bias = True)
            self.enc_out2 = nn.Linear(d_model_reduced, 1, bias = True)
        elif mod == 'dec':
            if user == 1:
                self.dec_out = nn.Linear(d_model, 2**args.K1, bias = True)
            elif user == 2:
                self.dec_out = nn.Linear(d_model, 2**args.K2, bias = True)
            else:
                raise ValueError('Invalid user')
        else:
            raise ValueError('Invalid mode')


    def forward(self, src):
        """
        input size = [batch_size, K]
        """
        src = src.float()
        # feature extractor
        feature_output = self.fe(src)
        norm_output = self.norm(feature_output)
        # MLP
        if self.mod == 'enc':
            mlp_output1 = self.enc_out1(norm_output)
            mlp_output = self.enc_out2(mlp_output1)
            return mlp_output
        else:
            mlp_output = self.dec_out(norm_output)
            # 2**K classes
            softmax_output = F.softmax(mlp_output, dim=-1)
            return softmax_output

class Power_reallocate(torch.nn.Module):
    def __init__(self, args):
        super(Power_reallocate, self).__init__()
        self.args = args
        self.weight = torch.nn.Parameter(torch.Tensor(self.args.N, 1), requires_grad=True)
        self.weight.data.uniform_(1.0, 1.0)
            
    def forward(self, inputs, idx):
       # symbol-level power allocation
        self.wt = torch.sqrt(self.weight ** 2 * (self.args.N / torch.sum(self.weight ** 2)))
        inputs1 = inputs * self.wt[idx] 
        return inputs1

class BC(nn.Module):
    def __init__(self, args):
        super(BC, self).__init__()
        self.args = args
        # encoder
        self.enc = MLP(args, 'enc', args.enc_input_size, args.enc_hidden_size)
    
        # decoder
        self.dec1 = MLP(args, 'dec', args.N-1, args.dec_hidden_size, user=1)
        self.dec2 = MLP(args, 'dec', args.N-1, args.dec_hidden_size, user=2)

        if args.power_allocation:
            self.power_allocation = Power_reallocate(args)

    def power_constraint(self, inputs):
        this_mean = torch.mean(inputs, 0)
        this_std  = torch.std(inputs, 0)
        outputs = (inputs - this_mean)*1.0/ (this_std + 1e-8)
        return outputs

    def forward(self, message1, forward_noise1, feedback_noise1, message2, forward_noise2, feedback_noise2):
        """
        message1: (num_samples, K1)
        forward_noise1: (num_samples, N)
        feedback_noise1: (num_samples, N)
        """
        num_samples_input = message1.shape[0]
        device = message1.device

        for idx in range(self.args.N):
        
            if idx == 0:
                message1_index, theta1 = PAMmodulation(message1, self.args.K1)
                norm_output = normalize(theta1, torch.tensor(1))
            elif idx == 1:
                message2_index, theta2 = PAMmodulation(message2, self.args.K2)
                norm_output = normalize(theta2, torch.tensor(1))
            else:
                if idx == self.args.N-1:
                    src = torch.cat([codewords, codewords_fb1, codewords_fb2], dim = 1)
                else:
                    src = torch.cat([codewords, torch.zeros(num_samples_input, (self.args.N - (idx+1))).to(device), codewords_fb1, torch.zeros(num_samples_input, (self.args.N - (idx+1))).to(device), codewords_fb2, torch.zeros(num_samples_input, (self.args.N - (idx+1))).to(device)], dim = 1)

                enc_output = self.enc(src)

                # encoder
                norm_output = self.power_constraint(enc_output)
            if self.args.power_allocation:
                norm_output = self.power_allocation(norm_output, idx)

            if idx == 0:
                codewords = norm_output
                codewords_rec1 = norm_output + forward_noise1[:, idx].unsqueeze(-1)
                codewords_rec2 = norm_output + forward_noise2[:, idx].unsqueeze(-1)

                codewords_fb1 = norm_output + forward_noise1[:, idx].unsqueeze(-1) + feedback_noise1[:, idx].unsqueeze(-1)
                codewords_fb2 = norm_output + forward_noise2[:, idx].unsqueeze(-1) + feedback_noise2[:, idx].unsqueeze(-1)
                
            else:
                codewords = torch.cat([codewords, norm_output], dim = 1)

                codewords_rec1 = torch.cat([codewords_rec1, norm_output + forward_noise1[:, idx].unsqueeze(-1)], dim = 1)
                codewords_rec2 = torch.cat([codewords_rec2, norm_output + forward_noise2[:, idx].unsqueeze(-1)], dim = 1)
                codewords_fb1 = torch.cat([codewords_fb1, norm_output + forward_noise1[:, idx].unsqueeze(-1) + feedback_noise1[:, idx].unsqueeze(-1)], dim = 1)
                codewords_fb2 = torch.cat([codewords_fb2, norm_output + forward_noise2[:, idx].unsqueeze(-1) + feedback_noise2[:, idx].unsqueeze(-1)], dim = 1)


        noisy_codewords1 = torch.cat([codewords_rec1[:,0:1], codewords_rec1[:,2:]], dim = 1)
        noisy_codewords2 = codewords_rec2[:,1:]
        # decoder
        dec_output1 = self.dec1(noisy_codewords1)
        dec_output2 = self.dec2(noisy_codewords2)
        return dec_output1, dec_output2, codewords, message1_index, message2_index

def errors_ber(y_true, y_pred, device):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    compare_result = torch.ne(y_true, y_pred).float()  
    res = torch.sum(compare_result)/(compare_result.shape[0] * compare_result.shape[1])  
    return res

def train_model(args, model, device):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    for eachbatch in range(args.totalbatch):
        # curriculum learning
        curriculum_nums = 1000
        if eachbatch < curriculum_nums:
            train_forward_snr= 3 * (1 - eachbatch/curriculum_nums)+ (eachbatch/ curriculum_nums) * args.forward_SNR1
        else:
            train_forward_snr = args.forward_SNR1

        # generate data
        message1, forward_noise1, feedback_noise1 = generate_data(args.batch_size, args.K1, args.N, train_forward_snr, args.feedback_SNR1)
        message2, forward_noise2, feedback_noise2 = generate_data(args.batch_size, args.K2, args.N, train_forward_snr, args.feedback_SNR2)

        dec_output1, dec_output2, codewords, message1_index, message2_index = model(message1.to(device), forward_noise1.to(device), feedback_noise1.to(device), message2.to(device), forward_noise2.to(device), feedback_noise2.to(device))

        args.optimizer.zero_grad()

        # true message
        message1_index = message1_index.long().contiguous().view(-1)
        message2_index = message2_index.long().contiguous().view(-1)

        # predicted message
        message1_preds = dec_output1.contiguous().view(-1, dec_output1.size(-1))
        message2_preds = dec_output2.contiguous().view(-1, dec_output2.size(-1))

        message1_preds_log = torch.log(message1_preds)
        message2_preds_log = torch.log(message2_preds)

        # loss
        loss1 = F.nll_loss(message1_preds_log, message1_index.to(device))
        loss2 = F.nll_loss(message2_preds_log, message2_index.to(device))
        lambda1 = 1
        loss = loss1 + loss2 + lambda1 * torch.square(loss1 - loss2)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        args.optimizer.step()
        args.scheduler.step()
        
        # Observe test accuracy 
        with torch.no_grad():
            probs1, decoded_index1 = message1_preds.max(dim=1)
            probs2, decoded_index2 = message2_preds.max(dim=1)
            succRate1 = sum(decoded_index1 == message1_index.to(device)) / message1_index.shape[0]
            succRate2 = sum(decoded_index2 == message2_index.to(device)) / message2_index.shape[0]
            message1_pred_binary = dec2bin(decoded_index1.view(-1, 1), args.K1)
            message2_pred_binary = dec2bin(decoded_index2.view(-1, 1), args.K2)
            ber1 = errors_ber(message1, message1_pred_binary, device)
            ber2 = errors_ber(message2, message2_pred_binary, device)   
            print(f"train_forward_snr: {train_forward_snr}, Idx: {eachbatch}, loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}, succRate1: {succRate1.item():.4f}, succRate2: {succRate2.item():.4f}, ber1: {ber1.item():.4f}, ber2: {ber2.item():.4f}")
        
        if eachbatch % 100 == 0 and eachbatch > 600:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            file_name = 'weights/model_'+date.today().strftime("%Y%m%d")+'_'+identity+'.pt'
            torch.save(model.state_dict(), file_name)
            print('final saved file_name = ', file_name)

    file_name = 'weights/model_'+date.today().strftime("%Y%m%d")+'_'+identity+'.pt'
    torch.save(model.state_dict(), file_name)
    print('final saved file_name = ', file_name)


    
def test(args, model, device, message1, forward_noise1, feedback_noise1, message2, forward_noise2, feedback_noise2):

    model.eval()

    dec_output1, dec_output2, codewords, message1_index, message2_index = model(message1.to(device), forward_noise1.to(device), feedback_noise1.to(device), message2.to(device), forward_noise2.to(device), feedback_noise2.to(device))

    print('----------power--------')
    print('codewords1 with mean:  ', torch.mean(codewords).cpu().detach().numpy())
    print('codewords1 with power: ', torch.var(codewords).cpu().detach().numpy())

    # true message
    message1_index = message1_index.long().contiguous().view(-1)
    message2_index = message2_index.long().contiguous().view(-1)

    # predicted message index
    message1_preds = dec_output1.contiguous().view(-1, dec_output1.size(-1))
    message2_preds = dec_output2.contiguous().view(-1, dec_output2.size(-1))

    # loss
    loss1 = F.nll_loss(torch.log(message1_preds), message1_index.to(device))
    loss2 = F.nll_loss(torch.log(message2_preds), message2_index.to(device))
    loss = loss1 + loss2

    # predicted message binary
    probs1, decoded_index1 = message1_preds.max(dim=1)
    probs2, decoded_index2 = message2_preds.max(dim=1)

    succRate1 = sum(decoded_index1 == message1_index.to(device)) / message1_index.shape[0]
    succRate2 = sum(decoded_index2 == message2_index.to(device)) / message2_index.shape[0]
    bler_1 = 1 - succRate1
    bler_2 = 1 - succRate2


    message1_pred_binary = dec2bin(decoded_index1.view(-1, 1), args.K1)
    message2_pred_binary = dec2bin(decoded_index2.view(-1, 1), args.K2)

    # calculate BER
    ber1 = errors_ber(message1, message1_pred_binary, device)
    ber2 = errors_ber(message2, message2_pred_binary, device)

    return loss1.item(), loss2.item(),  ber1.item(), ber2.item(), bler_1.item(), bler_2.item(), codewords


args = get_args()
print('args = ', args.__dict__)

use_cuda = args.with_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('use_cuda = ', use_cuda)
print('device = ', device)

if use_cuda:
    model = BC(args).to(device)
else:
    model = BC(args)
print(model)

args.initial_weights = 'rate39/light_bc_rate39_snr2.pt'
if args.initial_weights == 'default':
    pass
elif args.initial_weights == 'deepcode':
    f_load_deepcode_weights(model)
    print('deepcode weights are loaded.')
else:
    model.load_state_dict(torch.load(args.initial_weights, map_location=torch.device('cpu')))
    model.args = args
    print('initial weights are loaded.')


args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
lambda1 = lambda epoch: (1-epoch/args.totalbatch)
args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)


message1, forward_noise1, feedback_noise1 = generate_data(args.num_samples_test, args.K1, args.N, args.forward_SNR1, args.feedback_SNR1)
message2, forward_noise2, feedback_noise2 = generate_data(args.num_samples_test, args.K2, args.N, args.forward_SNR2, args.feedback_SNR2)

loss1, loss2,  ber1, ber2, bler1, bler2, codewords = test(args, model, device, message1, forward_noise1, feedback_noise1, message2, forward_noise2, feedback_noise2)
print('----- Ber 1(initial): ', ber1)
print('----- Ber 2(initial): ', ber2)
print('----- Bler 1: ', bler1)
print('----- Bler 2: ', bler2)
print(f'-----  loss1 (initial): {loss1}, loss2 (initial): {loss2}')

print("total power", torch.var(codewords).cpu().detach().numpy())
print("total mean", torch.mean(codewords).cpu().detach().numpy())


# for epoch in range(1, args.num_epoch + 1):
#     train_model(args, model, device)
#     loss1, loss2,  ber1, ber2, codewords = test(args, model, device, message1, forward_noise1, feedback_noise1, message2, forward_noise2, feedback_noise2)
#     print('----- epoch: ', epoch)
#     print('----- Ber 1: ', ber1)
#     print('----- Ber 2: ', ber2)
#     print(f'-----  loss1: {loss1}, loss2: {loss2}')

# nums = 100
# ber1_list = []
# ber2_list = []
# bler1_list = []
# bler2_list = []
# for epoch in range(1, nums + 1):
#     print(f"-----------------------{epoch}------------------------")
#     message1, forward_noise1, feedback_noise1 = generate_data(args.num_samples_test, args.K1, args.N, args.forward_SNR1, args.feedback_SNR1)
#     message2, forward_noise2, feedback_noise2 = generate_data(args.num_samples_test, args.K2, args.N, args.forward_SNR2, args.feedback_SNR2)
#     loss1, loss2,  ber1, ber2, bler1, bler2, codewords = test(args, model, device, message1, forward_noise1, feedback_noise1, message2, forward_noise2, feedback_noise2)
#     print('----- epoch: ', epoch)
#     print('----- Ber 1: ', ber1)
#     print('----- Ber 2: ', ber2)
#     print('----- Bler 1: ', bler1)
#     print('----- Bler 2: ', bler2)
#     print(f'-----  loss1: {loss1}, loss2: {loss2}')
#     ber1_list.append(ber1)
#     ber2_list.append(ber2)
#     bler1_list.append(bler1)
#     bler2_list.append(bler2)
# avg_ber1 = sum(ber1_list)/nums
# avg_ber2 = sum(ber2_list)/nums

# avg_bler1 = sum(bler1_list)/nums
# avg_bler2 = sum(bler2_list)/nums
# print('----- avg_ber1: ', avg_ber1)
# print('----- avg_ber2: ', avg_ber2)
# print('----- avg_ber: ', (avg_ber1 + avg_ber2)/2)
# print('----- avg_bler1: ', avg_bler1)
# print('----- avg_bler2: ', avg_bler2)
# print('----- avg_bler: ', (avg_bler1 + avg_bler2)/2)

       

