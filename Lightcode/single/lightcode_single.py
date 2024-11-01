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

# identity = 'lightcode_single_rate34_snr1'
# print('[ID]', identity)


def get_args():
    parser = argparse.ArgumentParser()

    # channel configuration
    parser.add_argument('-forward_SNR1', type=int, default=3)
    parser.add_argument('-feedback_SNR1', type=int, default=100, help='100 means noiseless feeback')  

    parser.add_argument('-K1', type=int, default=3)
    parser.add_argument('-N', type=int, default=5, help='Number of transmission rounds')

    parser.add_argument('-enc_input_size', type=int, default=11) # K1 + 2(N-1)
    parser.add_argument('-enc_hidden_size', type=int, default=16)
    parser.add_argument('-dec_hidden_size', type=int, default=16)
    parser.add_argument('-power_allocation', type=bool, default=True)
    parser.add_argument('-num_samples_test', type=int, default=100000)


    # training configuration
    parser.add_argument('-totalbatch', type=int, default=4000, help="number of total batches to train")
    parser.add_argument('-num_epoch', type=int, default=1, help="number of epochs to train")
    parser.add_argument('-batch_size', type=int, default=100000, help="batch size") # snr is high, use larger batch size
    parser.add_argument('-lr', type=float, default=0.005, help="learning rate") # 0.002 further training use smaller lr
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



class FE(nn.Module):
    def __init__(self, args, mod, input_size, d_model):
        super(FE, self).__init__()
        self.args = args
        self.mod = mod

        if mod == 'enc':
            self.activation = nn.ReLU()
            self.FC1 = nn.Linear(input_size, 2 * d_model, bias=True)
            self.FC2 = nn.Linear(2 * d_model, 2 * d_model, bias=True)
            self.FC3 = nn.Linear(2 * d_model, 2 * d_model, bias=True)
            self.FC4 = nn.Linear(4 * d_model, d_model, bias = True)

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
            x1 = self.FC1(src)
            x2 = self.FC2(self.activation(x1))
            x3 = self.FC3(self.activation(x2))
            x = self.FC4(torch.cat([x3, -1 * x1], dim = 1))
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
        if mod == 'enc':
            # encoder output
            d_model_reduced = int(d_model/4)
            self.enc_out1 = nn.Linear(d_model, d_model_reduced, bias = True)
            self.enc_out2 = nn.Linear(d_model_reduced, 1, bias = True)
        elif mod == 'dec':
            # decoder output
            self.dec_out = nn.Linear(d_model, 2**args.K1, bias = True)
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
        self.dec1 = MLP(args, 'dec', args.N, args.dec_hidden_size)

        if args.power_allocation:
            self.power_allocation = Power_reallocate(args)

    def power_constraint(self, inputs):
        this_mean = torch.mean(inputs, 0)
        this_std  = torch.std(inputs, 0)
        outputs = (inputs - this_mean)*1.0/ (this_std + 1e-8)
        return outputs

    def forward(self, message1, forward_noise1, feedback_noise1):
        """
        message1: (num_samples, K1)
        forward_noise1: (num_samples, N)
        feedback_noise1: (num_samples, N)
        """
        num_samples_input = message1.shape[0]
        device = message1.device
        message1_mod = 2 * message1 - 1
        
        for idx in range(self.args.N):
            noises = forward_noise1 + feedback_noise1
            # input y feedback
            if idx == 0:
                src1 = torch.cat([message1_mod, torch.zeros(num_samples_input, 2 * (self.args.N - 1)).to(device)], dim = 1)
            elif idx == self.args.N-1:
                src1 = torch.cat([message1_mod, codewords, codewords_fb1], dim=1)
            else:
                src1 = torch.cat([message1_mod, codewords, torch.zeros(num_samples_input, (self.args.N - (idx+1))).to(device), codewords_fb1, torch.zeros(num_samples_input, (self.args.N - (idx+1))).to(device)], dim = 1)
            # encoder
            enc_output = self.enc(src1)
            norm_output = self.power_constraint(enc_output)
            if self.args.power_allocation:
                norm_output = self.power_allocation(norm_output, idx)

            if idx == 0:
                codewords_fb1 = norm_output + forward_noise1[:, idx].unsqueeze(-1) + feedback_noise1[:, idx].unsqueeze(-1)
                codewords = norm_output
            else:
                codewords_fb1 = torch.cat([codewords_fb1, norm_output + forward_noise1[:, idx].unsqueeze(-1) + feedback_noise1[:, idx].unsqueeze(-1)], dim = 1)
                codewords = torch.cat([codewords, norm_output], dim = 1)

        noisy_codewords1 = codewords + forward_noise1
        dec_output1 = self.dec1(noisy_codewords1)
        return dec_output1, codewords

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
        # # curriculum learning
        # curriculum_nums = 1000
        # if eachbatch < curriculum_nums:
        #     train_forward_snr= 4 * (1 - eachbatch/curriculum_nums)+ (eachbatch/curriculum_nums) * args.forward_SNR1
        # else:
        #     train_forward_snr = args.forward_SNR1
        train_forward_snr = args.forward_SNR1

        # generate data
        message1, forward_noise1, feedback_noise1 = generate_data(args.batch_size, args.K1, args.N, train_forward_snr, args.feedback_SNR1)


        dec_output1, codewords1 = model(message1.to(device), forward_noise1.to(device), feedback_noise1.to(device))
        # print('----------power--------')
        # print('codewords1 with mean:  ', torch.mean(codewords1).cpu().detach().numpy())
        # print('codewords1 with power: ', torch.var(codewords1).cpu().detach().numpy())

        args.optimizer.zero_grad()

        # true message
        message1_index = bin2dec(message1, args.K1)
        message1_index = message1_index.long().contiguous().view(-1)
        

        # predicted message
        message1_preds = dec_output1.contiguous().view(-1, dec_output1.size(-1))
        message1_preds_log = torch.log(message1_preds)
      

        # loss
        loss = F.nll_loss(message1_preds_log, message1_index.to(device))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        args.optimizer.step()
        args.scheduler.step()

        # Observe test accuracy 
        with torch.no_grad():
            probs1, decoded_index1 = message1_preds.max(dim=1)
            succRate1 = sum(decoded_index1 == message1_index.to(device)) / message1_index.shape[0]
            message1_pred_binary = dec2bin(decoded_index1.view(-1, 1), args.K1)
            ber1 = errors_ber(message1, message1_pred_binary, device)
            bler1 = 1 - succRate1
            print(f"Idx: {eachbatch}, loss1: {loss.item():.4f}, succRate1: {succRate1.item():.4f}, ber1: {ber1.item():.4f}, bler1: {bler1.item():.4f}")
        
        if eachbatch % 100 == 0 and eachbatch > 400:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            file_name = 'weights/model_'+ identity +'_'+str(eachbatch)+'.pt'
            torch.save(model.state_dict(), file_name)
            print('final saved file_name = ', file_name)

    file_name = 'weights/model_'+date.today().strftime("%Y%m%d")+'_'+identity+'.pt'
    torch.save(model.state_dict(), file_name)
    print('final saved file_name = ', file_name)
    
def test(args, model, device, message1, forward_noise1, feedback_noise1):

    model.eval()

    dec_output1, codewords1 = model(message1.to(device), forward_noise1.to(device), feedback_noise1.to(device))

    print('----------power--------')
    print('codewords1 with mean:  ', torch.mean(codewords1).cpu().detach().numpy())
    print('codewords1 with power: ', torch.var(codewords1).cpu().detach().numpy())

    # true message
    message1_index = bin2dec(message1, args.K1)

    message1_index = message1_index.long().contiguous().view(-1)

    # predicted message index
    message1_preds = dec_output1.contiguous().view(-1, dec_output1.size(-1))

    # loss
    loss1 = F.nll_loss(torch.log(message1_preds), message1_index.to(device))
    
    # predicted message binary
    probs1, decoded_index1 = message1_preds.max(dim=1)
    succRate1 = sum(decoded_index1 == message1_index.to(device)) / message1_index.shape[0]
    bler1 = 1 - succRate1

    message1_pred_binary = dec2bin(decoded_index1.view(-1, 1), args.K1)

    # calculate BER
    ber1 = errors_ber(message1, message1_pred_binary, device)

    return loss1, ber1.item(), bler1.item(), codewords1


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

args.initial_weights = 'rate35/lightcode_single_rate35_snr3.pt'
if args.initial_weights == 'default':
    pass
else:
    model.load_state_dict(torch.load(args.initial_weights, map_location=torch.device('cpu')))
    model.args = args
    print('initial weights are loaded.')


args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
lambda1 = lambda epoch: (1-epoch/args.totalbatch)
args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)

message1, forward_noise1, feedback_noise1 = generate_data(args.num_samples_test, args.K1, args.N, args.forward_SNR1, args.feedback_SNR1)

loss1,  ber1, bler1, codewords1 = test(args, model, device, message1, forward_noise1, feedback_noise1)
print('----- Ber 1(initial): ', ber1)
print('----- Bler 1(initial): ', bler1)
print(f'-----  loss1 (initial): {loss1}')


# for epoch in range(1, args.num_epoch + 1):
#     train_model(args, model, device)
#     loss1,  ber1, bler1, codewords1 = test(args, model, device, message1, forward_noise1, feedback_noise1)
#     print('----- Ber 1: ', ber1)
#     print('----- Bler 1(initial): ', bler1)
#     print(f'-----  loss1: {loss1}')

# nums = 100
# ber1_list = []
# bler1_list = []

# for epoch in range(1, nums + 1):
#     print(f"-----------------------{epoch}------------------------")
#     message1, forward_noise1, feedback_noise1 = generate_data(args.num_samples_test, args.K1, args.N, args.forward_SNR1, args.feedback_SNR1)
#     loss1,  ber1, bler1, codewords1 = test(args, model, device, message1, forward_noise1, feedback_noise1)
#     print('----- Ber 1: ', ber1)
#     print('----- Bler 1: ', bler1)
#     print(f'-----  loss1: {loss1}')
#     ber1_list.append(ber1)
#     bler1_list.append(bler1)
# avg_ber1 = sum(ber1_list)/nums
# avg_bler1 = sum(bler1_list)/nums
# print('----- avg_ber1: ', avg_ber1)
# print('----- avg_bler1: ', avg_bler1)
       
       

