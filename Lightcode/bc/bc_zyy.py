import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import *
from Feature_extractors_bc import FE
import copy
from parameters_bc import *
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm

def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

class ae_backbone(nn.Module):
    def __init__(self, arch, mod, input_size, m, d_model, dropout, multclass = False, NS_model=0):
        super(ae_backbone, self).__init__()
        self.arch = arch
        self.mod = mod
        self.multclass = multclass
        self.m = m
        self.relu = nn.ReLU()

        self.fe1 = FE(mod, NS_model, input_size, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
    
        if mod == "trx":
            self.out1 = nn.Linear(d_model, d_model)
            self.out2 = nn.Linear(d_model, 1)
        else:
            if multclass:
                self.out = nn.Linear(d_model, 2**m)
            else:
                self.out = nn.Linear(d_model, 2*m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        enc_out = self.fe1(src)
        enc_out = self.norm1(enc_out)
        
        if self.mod == "rec":
            enc_out = self.out(self.relu(enc_out))
        else:
            enc_out = self.out1(self.relu(enc_out))
            enc_out = self.out2(self.relu(enc_out))
   
        if self.mod == "rec":
            if self.multclass == False:
                batch = enc_out.size(0)
                ell = enc_out.size(1)
                enc_out = enc_out.contiguous().view(batch, ell*self.m,2)
                output = F.softmax(enc_out, dim=-1)
            else:
                output = F.softmax(enc_out, dim=-1)
        else:
            output = enc_out
        return output

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




########################## This is the overall AutoEncoder model ########################

class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        self.Tmodel = ae_backbone(args.arch, "trx",  3*(self.args.T-1), args.m, args.d_model_trx, args.dropout, args.multclass, args.enc_NS_model)
        self.Rmodel1 = ae_backbone(args.arch, "rec", args.T, args.m, args.d_model_rec, args.dropout, args.multclass, args.dec_NS_model)
        self.Rmodel2 = ae_backbone(args.arch, "rec", args.T, args.m, args.d_model_rec, args.dropout, args.multclass, args.dec_NS_model)
        
        #Power Reallocation 
        if self.args.reloc == 1:
            self.total_power_reloc = Power_reallocate(args)

    def power_constraint(self, inputs, isTraining, train_mean, train_std, idx = 0): # Normalize through batch dimension

        if isTraining == 1 or train_mean is None:
            # training
            this_mean = torch.mean(inputs, 0)   
            this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            # use stats from training
            this_mean = train_mean[idx]
            this_std = train_std[idx]

        outputs = (inputs - this_mean)*1.0/ (this_std + 1e-8)
  
        return outputs, this_mean.detach(), this_std.detach()

    def forward(self, train_mean, train_std, bVec_1, fwd_noise_par1, fb_noise_par1, bVec_2, fwd_noise_par2, fb_noise_par2, isTraining = 1):
        combined_noise_par1 = fwd_noise_par1 + fb_noise_par1 
        combined_noise_par2 = fwd_noise_par2 + fb_noise_par2
        for idx in range(self.args.T): 
            if idx == 0:
                message1_index, theta1 = PAMmodulation(bVec_1, self.args.m)
                output = theta1
            elif idx == 1:
                message2_index, theta2 = PAMmodulation(bVec_2, self.args.m)
                output = theta2
            else:
                if idx == self.args.T-1:
                	src = torch.cat([parity_all, parity_fb1, parity_fb2],dim=2)
                else:
                	src = torch.cat([parity_all, torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device), parity_fb1, torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device), parity_fb2, torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device)],dim=2)
                output = self.Tmodel(src)
            
            parity, x_mean, x_std = self.power_constraint(output, isTraining, train_mean, train_std, idx)

            if self.args.reloc == 1:
                parity = self.total_power_reloc(parity,idx)

            if idx == 0:
                parity_fb1 = parity + combined_noise_par1[:,:,idx].unsqueeze(-1)
                parity_fb2 = parity + combined_noise_par2[:,:,idx].unsqueeze(-1)
                parity_all = parity
                received1 = parity + fwd_noise_par1[:,:,0].unsqueeze(-1)
                received2 = parity + fwd_noise_par2[:,:,0].unsqueeze(-1)
                x_mean_total, x_std_total = x_mean, x_std
            else:
                parity_fb1 = torch.cat([parity_fb1, parity + combined_noise_par1[:,:,idx].unsqueeze(-1)],dim=2) 
                parity_fb2 = torch.cat([parity_fb2, parity + combined_noise_par2[:,:,idx].unsqueeze(-1)],dim=2) 
                parity_all = torch.cat([parity_all, parity], dim=2)     
                received1 = torch.cat([received1, parity + fwd_noise_par1[:,:,idx].unsqueeze(-1)], dim = 2)
                received2 = torch.cat([received2, parity + fwd_noise_par2[:,:,idx].unsqueeze(-1)], dim = 2)

                x_mean_total = torch.cat([x_mean_total, x_mean], dim = 0)
                x_std_total = torch.cat([x_std_total, x_std], dim = 0)

        # Decoding
        # print("->-->-->-->-->-->-->-->-->-->  power", torch.mean(torch.square(parity_all)))

        decSeq1 = self.Rmodel1(received1) 
        decSeq2 = self.Rmodel2(received2) 
        
        return decSeq1, decSeq2, x_mean_total, x_std_total, parity_all, message1_index, message2_index

def train_model(model, args, logging):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    logging.info("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    start = time.time()
    epoch_loss_record = []
    
    pbar = tqdm(range(args.totalbatch))
    train_mean = torch.zeros(args.T, 1).to(args.device)
    train_std = torch.zeros(args.T, 1).to(args.device)
    
    pktErrors1 = 0
    pktErrors2 = 0
    for eachbatch in pbar:
        bVec_1 = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
        bVec_2 = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
        
      
        snr2=args.snr2
        if eachbatch < 0:
            snr1=4* (1-eachbatch/(args.core * 30000))+ (eachbatch/(args.core * 30000)) * args.snr1
        else:
            snr1=args.snr1
        ################################################################################################################
        std1 = 10 ** (-snr1 * 1.0 / 10 / 2) #forward snr
        std2 = 10 ** (-snr2 * 1.0 / 10 / 2) #feedback snr
        # Noise values for the parity bits
        fwd_noise_par1 = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_par1 = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fwd_noise_par2 = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_par2 = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        if args.snr2 >= 100:
            fb_noise_par1 = 0* fb_noise_par1
            fb_noise_par2 = 0* fb_noise_par2
        if np.mod(eachbatch, args.core) == 0:
            w_locals = []
            w0 = model.state_dict()
            w0 = copy.deepcopy(w0)
        else:
            # Use the common model to have a large batch strategy
            model.load_state_dict(w0)

        # feed into model to get predictions
        preds1, preds2, batch_mean, batch_std, parity_all, message1_index, message2_index = model(None, None, bVec_1.to(args.device), fwd_noise_par1.to(args.device), fb_noise_par1.to(args.device), bVec_2.to(args.device), fwd_noise_par2.to(args.device), fb_noise_par2.to(args.device), isTraining=1)


        if batch_mean is not None:
            train_mean += batch_mean
            train_std += batch_std # not the best way but numerically stable

        args.optimizer.zero_grad()

        ys1 = message1_index.long().contiguous().view(-1)
        preds1 = preds1.contiguous().view(-1, preds1.size(-1)) #=> (Batch*K) x 2
        preds1 = torch.log(preds1)
        loss1 = F.nll_loss(preds1, ys1.to(args.device))

        ys2 = message2_index.long().contiguous().view(-1)
        preds2 = preds2.contiguous().view(-1, preds1.size(-1)) #=> (Batch*K) x 2
        preds2 = torch.log(preds2)
        loss2 = F.nll_loss(preds2, ys2.to(args.device))


        probs1, decodeds1 = preds1.max(dim=1)
        decisions1 = decodeds1 != ys1.to(args.device)
        probs2, decodeds2 = preds2.max(dim=1)
        decisions2 = decodeds2 != ys2.to(args.device)

        pktErrors1 += decisions1.view(args.batchSize, args.ell).sum(1).count_nonzero()
        PER1 = pktErrors1 / (eachbatch + 1) / args.batchSize
        pktErrors2 += decisions2.view(args.batchSize, args.ell).sum(1).count_nonzero()
        PER2 = pktErrors2 / (eachbatch + 1) / args.batchSize

        loss = loss1 + loss2 + torch.square(loss1 - loss2)


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        args.optimizer.step()
        # Save the model
        w1 = model.state_dict()
        w_locals.append(copy.deepcopy(w1))
        if np.mod(eachbatch, args.core) != args.core - 1:
            continue
        else:
            w2 = ModelAvg(w_locals) # Average the models
            model.load_state_dict(copy.deepcopy(w2))
            if args.use_lr_schedule:
                args.scheduler.step()
        if eachbatch%10000 == 0:
            with torch.no_grad():
                print(f"\nGBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 {snr1}, snr2 {snr2}, BS {args.batchSize}, Loss {round(loss.item(), 8)}, PER1 {round(PER1.item(), 10)}, PER2 {round(PER2.item(), 10)}")
                logging.info(f"\nGBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 {snr1}, snr2 {snr2}, BS {args.batchSize}, Loss {round(loss.item(), 8)}, PER1 {round(PER1.item(), 10)}, PER2 {round(PER2.item(), 10)}")		
   
                print("Testing started: ... ")
                logging.info("Testing started: ... ")
                # change batch size to 10x for testing
                args.batchSize = int(args.batchSize*10)
                EvaluateNets(model, None, None, args, logging)
                args.batchSize = int(args.batchSize/10)
                print("... finished testing")
    
        if np.mod(eachbatch, args.core * 10000) == args.core - 1:
            epoch_loss_record.append(loss.item())
            if not os.path.exists(weights_folder):
                os.mkdir(weights_folder)
            torch.save(epoch_loss_record, f'{weights_folder}/loss')

        if np.mod(eachbatch, args.core * 10000) == args.core - 1:# and eachbatch >= 80000:
            if not os.path.exists(weights_folder):
                os.mkdir(weights_folder)
            saveDir = f'{weights_folder}/model_weights' + str(eachbatch) + '.pt'
            torch.save(model.state_dict(), saveDir)
        pbar.update(1)
        pbar.set_description(f"GBAF train stats: batch#{eachbatch}, Loss {round(loss.item(), 8)}")

        # kill the training if the loss is nan
        if np.isnan(loss.item()):
            print("Loss is nan, killing the training")
            logging.info("Loss is nan, killing the training")
            break
  
    pbar.close()

    if train_mean is not None:
        train_mean = train_mean / args.totalbatch
        train_std = train_std / args.totalbatch	# not the best way but numerically stable
      
    return train_mean, train_std

def EvaluateNets(model, train_mean, train_std, args, logging):
    if args.train == 0:
        path = f'{weights_folder}/model_weights{args.totalbatch-101}.pt'
        print(f"Using model from {path}")
        logging.info(f"Using model from {path}")
    
        checkpoint = torch.load(path,map_location=args.device)
    
        #load weights
        model.load_state_dict(checkpoint)
        model = model.to(args.device)
    model.eval()
   

    args.numTestbatch = 100000000
    
    symErrors1 = 0
    symErrors2 = 0

 
    start_time = time.time()

    for eachbatch in range(args.numTestbatch):
        bVec_1 = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
        bVec_2 = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
        std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
        std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
        fwd_noise_par1 = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_par1 = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fwd_noise_par2 = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_par2 = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par1 = 0* fb_noise_par1
            fb_noise_par2 = 0* fb_noise_par2

        # feed into model to get predictions
        with torch.no_grad():
            preds1, preds2, _, _, parity_all, message1_index, message2_index = model(train_mean, train_std, bVec_1.to(args.device), fwd_noise_par1.to(args.device), fb_noise_par1.to(args.device), bVec_2.to(args.device), fwd_noise_par2.to(args.device), fb_noise_par2.to(args.device), isTraining=0)

            
            ys1 = message1_index.long().contiguous().view(-1)
            ys2 = message2_index.long().contiguous().view(-1)

        
            preds1 = preds1.contiguous().view(-1, preds1.size(-1))
            preds2 = preds2.contiguous().view(-1, preds1.size(-1))

            

            probs1, decodeds1 = preds1.max(dim=1)
            probs2, decodeds2 = preds2.max(dim=1)
        

            decisions1 = decodeds1 != ys1.to(args.device)
            decisions2 = decodeds2 != ys2.to(args.device)

            symErrors1 += decisions1.sum()
            SER1 = symErrors1 / (eachbatch + 1) / args.batchSize / args.ell
            symErrors2 += decisions2.sum()
            SER2 = symErrors2 / (eachbatch + 1) / args.batchSize / args.ell

            num_batches_ran = eachbatch + 1
            num_pkts = num_batches_ran * args.batchSize	

            if eachbatch%1000 == 0:
                print(f"\nGBAF test stats: batch#{eachbatch}, SER1 {round(SER1.item(), 10)}, numErr1 {symErrors1.item()}, num_pkts1 {num_pkts:.2e}")
                print(f"\nGBAF test stats: batch#{eachbatch}, SER2 {round(SER2.item(), 10)}, numErr2 {symErrors2.item()}, num_pkts2 {num_pkts:.2e}")
                logging.info(f"\nGBAF test stats: batch#{eachbatch}, SER2 {round(SER1.item(), 10)}, numErr {symErrors1.item()}, num_pkts {num_pkts:.2e}")
                logging.info(f"\nGBAF test stats: batch#{eachbatch}, SER2 {round(SER2.item(), 10)}, numErr {symErrors2.item()}, num_pkts {num_pkts:.2e}")
                print(f"Time elapsed: {(time.time() - start_time)/60} mins")
                logging.info(f"Time elapsed: {(time.time() - start_time)/60} mins")
            if args.train == 1:
                min_err = 20
            else:
                min_err = 100
            if symErrors1 > min_err or (args.train == 1 and num_batches_ran * args.batchSize * args.ell > 1e8):
                print(f"\nGBAF test stats: batch#{eachbatch}, SER {round(SER1.item(), 10)}, numErr {symErrors1.item()}")
                logging.info(f"\nGBAF test stats: batch#{eachbatch}, SER {round(SER1.item(), 10)}, numErr {symErrors1.item()}")
                break

            if symErrors2 > min_err or (args.train == 1 and num_batches_ran * args.batchSize * args.ell > 1e8):
                print(f"\nGBAF test stats: batch#{eachbatch}, SER {round(SER2.item(), 10)}, numErr {symErrors2.item()}")
                logging.info(f"\nGBAF test stats: batch#{eachbatch}, SER {round(SER2.item(), 10)}, numErr {symErrors2.item()}")
                break

    SER1 = symErrors1.cpu() / (num_batches_ran * args.batchSize * args.ell)
    print(f"Final test SER1 = {torch.mean(SER1).item()}, at SNR1 {args.snr1}, SNR2 {args.snr2} for rate {args.m}/{args.T}")
    logging.info(f"Final test SER1 = {torch.mean(SER1).item()}, at SNR1 {args.snr1}, SNR2 {args.snr2} for rate {args.m}/{args.T}")
    SER2 = symErrors2.cpu() / (num_batches_ran * args.batchSize * args.ell)
    print(f"Final test SER2 = {torch.mean(SER2).item()}, at SNR1 {args.snr1}, SNR2 {args.snr2} for rate {args.m}/{args.T}")
    print(f"average SER", (torch.mean(SER1).item() + torch.mean(SER2).item())/2)
    logging.info(f"Final test SER2 = {torch.mean(SER2).item()}, at SNR1 {args.snr1}, SNR2 {args.snr2} for rate {args.m}/{args.T}")
   




if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser(True)
    ########### path for saving model checkpoints ################################
    args.saveDir = f'weights/model_weights{args.totalbatch-101}.pt'  # path to be saved to
    args.d_model_trx = args.d_k_trx # total number of features
    args.d_model_rec = args.d_k_rec # total number of features
 
    # fix the random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    model = AE(args).to(args.device)
    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True
 
    model = AE(args).to(args.device)
  
    # configure the logging
    folder_str = f"T_{args.T}/pow_{args.reloc}/{args.batchSize}/{args.lr}/"
    sim_str = f"K_{args.K}_m_{args.m}_snr1_{args.snr1}"
 
    parent_folder = f"zyy_results/N_{args.enc_NS_model}_{args.dec_NS_model}_d_{args.d_k_trx}_{args.d_k_rec}/snr2_{args.snr2}/seed_{args.seed}"
 
    log_file = f"log_{sim_str}.txt"
    log_folder = f"{parent_folder}/logs/gbaf_{args.arch}_{args.features}/{folder_str}"
    log_file_name = os.path.join(log_folder, log_file)
 
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(format='%(message)s', filename=log_file_name, encoding='utf-8', level=logging.INFO)

    global weights_folder
    weights_folder = f"{parent_folder}/weights/gbaf_{args.arch}_{args.features}/{folder_str}/{sim_str}/"
    os.makedirs(weights_folder, exist_ok=True)


    if args.train == 1:
        if args.opt_method == 'adamW':
            args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        elif args.opt_method == 'lamb':
            args.optimizer = optim.Lamb(model.parameters(),lr= 1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        else:
            args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        if args.use_lr_schedule:
            lambda1 = lambda epoch: (1-epoch/args.totalbatch)
            args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)

        # print the model summary
        print(model)
        logging.info(model)
  
        # print the number of parameters in the model that need to be trained
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {num_params}")
        logging.info(f"Total number of trainable parameters: {num_params}")
  
        # print num params in Tmodel
        num_params = sum(p.numel() for p in model.Tmodel.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters in Tmodel: {num_params}")
        logging.info(f"Total number of trainable parameters in Tmodel: {num_params}")
        # print num params in Rmodel
        num_params = sum(p.numel() for p in model.Rmodel1.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters in Rmodel1: {num_params}")
        logging.info(f"Total number of trainable parameters in Rmodel1: {num_params}")


        train_mean, train_std = train_model(model, args, logging)

        # stop training and test
        args.train = 0
        args.batchSize = int(args.batchSize*10)
        start_time = time.time()
  
        print("\nInference after training: ... ")
        logging.info("\nInference after training: ... ")
        EvaluateNets(model, None, None, args, logging)
        args.batchSize = int(args.batchSize/10)
  
        end_time = time.time()
        tot_time_mins = (end_time - start_time) / 60
        print(f"\nTime for testing: {tot_time_mins}")
        logging.info(f"\nTime for testing: {tot_time_mins}")

    ## Inference
    print("\nInference using trained model and stats from large dataset: ... ")
    logging.info("\nInference using trained model and stats from large dataset: ... ")

    path = f'{weights_folder}/model_weights{args.totalbatch-101}.pt'
    print(f"\nUsing model from {path}")
    logging.info(f"\nUsing model from {path}")
 
    # use one very large batch to compute mean and std
    large_bs = int(1e6)
    args.batchSize = large_bs
    checkpoint = torch.load(path,map_location=args.device)

    # ======================================================= load weights
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    model.eval()
   

    
    bVec_1 = torch.randint(0, 2, (large_bs, args.ell, args.m))
    bVec_2 = torch.randint(0, 2, (large_bs, args.ell, args.m))
    std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
    std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
    fwd_noise_par1 = torch.normal(0, std=std1, size=(large_bs, args.ell, args.T), requires_grad=False)
    fb_noise_par1 = torch.normal(0, std=std2, size=(large_bs, args.ell, args.T), requires_grad=False)

    fwd_noise_par2 = torch.normal(0, std=std1, size=(large_bs, args.ell, args.T), requires_grad=False)
    fb_noise_par2 = torch.normal(0, std=std2, size=(large_bs, args.ell, args.T), requires_grad=False)
    if args.snr2 == 100:
        fb_noise_par1 = 0* fb_noise_par1
        fb_noise_par2 = 0* fb_noise_par2

    # feed into model to get predictions
    with torch.no_grad():
        preds1, preds2, train_mean, train_std, parity_all, message1_index, message2_index = model(None, None, bVec_1.to(args.device), fwd_noise_par1.to(args.device), fb_noise_par1.to(args.device), bVec_2.to(args.device), fwd_noise_par2.to(args.device), fb_noise_par2.to(args.device), isTraining=0)

    EvaluateNets(model, train_mean, train_std, args, logging)