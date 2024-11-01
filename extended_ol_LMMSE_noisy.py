# extend EOL scheme to noisy feedback case
import torch 
import numpy as np
import matplotlib.pyplot as plt 
import argparse

torch.set_default_dtype(torch.float64)
def get_args(jupyter_notebook):
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-power', type = int, default = 1)
	parser.add_argument('-g', type = int, default=1, help="nonnegative number allows a trade-off of achievable rates to receivers 1 and 2")  

	# message bits
	parser.add_argument('-K1', type = int, default=3) 
	parser.add_argument('-K2', type = int, default=3) 
	parser.add_argument('-N', type = int, default=9)  


	# channel definition
	parser.add_argument('-forward_SNR1', type=int, default=3) # forward SNR for receiver 1
	parser.add_argument('-forward_SNR2', type=int, default=3) # forward SNR for receiver 2
	parser.add_argument('-feedback_SNR1', type=int, default=30, help='100 means noiseless feeback') # feedback SNR for receiver 1
	parser.add_argument('-feedback_SNR2', type=int, default=30, help='100 means noiseless feeback') # feedback SNR for receiver 1

	parser.add_argument('-rhoz', type = int, default=0) # the correlation coefficient of the Gaussian noises

	parser.add_argument('-num_samples', type=int, default=100000) 
	if jupyter_notebook:
		args = parser.parse_args(args=[])   # for jupyter notebook
	else:
		args = parser.parse_args()    # in general
	return args

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

def sgn(data):
	"""
	if data >= 0, 1
	if data < 0, -1
	"""
	condition = (data >= 0)
	branch1 = torch.ones(data.shape)
	branch2 = - torch.ones(data.shape)
	res = torch.where(condition, branch1, branch2)
	return res
	
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


def snr_db_2_sigma(snr_db, feedback=False):
    if feedback and snr_db == 100:
        return 0
    return 10**(-snr_db * 1.0 / 20)

def generate_data(K, N, forward_SNR, feedback_SNR, num_samples):
	"""
	K: length of message bits
	N: number of rounds
	Output: 
	message (num_samples, K)
	forward_noise (num_samples, N)
	feedback_noise (num_samples, N)
	"""
	message = torch.randint(0, 2, (num_samples, K))
	forward_sigma = snr_db_2_sigma(forward_SNR)
	feedback_sigma = snr_db_2_sigma(feedback_SNR, feedback=True)
	forward_noise = forward_sigma * torch.randn((num_samples, N))
	feedback_noise = feedback_sigma * torch.randn((num_samples, N))
	return message, forward_noise, feedback_noise

def update(alpha1, alpha2, rho, lambda1, lambda2, P, g, sigma1, sigma2, rhoz,  pi1, pi2, psi_kminus1):
	"""
	Update the alpha 1, alpha2, and rho
	"""

	alpha1_next = get_alpha(alpha1, pi1, lambda1, P, psi_kminus1, g, rho, 1)
	alpha2_next = get_alpha(alpha2, pi2, lambda2, P, psi_kminus1, g, rho, 2)

	phi = torch.square(psi_kminus1) * (g + torch.abs(rho)) * (1 + g * torch.abs(rho)) * sgn(rho)
	Sigma = P + sigma1**2 + sigma2**2 - rhoz * sigma1 * sigma2
	T = rho * g * pi1**2 * pi2**2 - g * phi * pi1 * pi2 * Sigma + torch.square(lambda1) * pi2**2 * sgn(rho) + g**2 * torch.square(lambda2) * pi1**2 * sgn(rho) - torch.square(lambda1) * torch.square(lambda2) * sgn(rho) * (1 + g**2 + 2 * g * sgn(rho)) + g * phi * lambda1 * lambda2 * (P + rhoz * sigma1 * sigma2)
	Omega = g * torch.sqrt((pi1**2 - torch.square(lambda1)) * (pi2**2 - torch.square(lambda2))) * torch.sqrt(pi1**2 - torch.square(lambda1) - P * pi1 + torch.square(psi_kminus1) * g**2 * (1 - torch.square(rho)) * pi1) * torch.sqrt(pi2**2 - torch.square(lambda2) - P * pi2 + torch.square(psi_kminus1) * (1 - torch.square(rho)) * pi2)

	rho_next = T / Omega
	return alpha1_next, alpha2_next, rho_next
	

def get_alpha(alpha, pi, lambdak, P, psi, g, rho, i):
	"""
	i is the user
	"""
	num = pi**2 - torch.square(lambdak) - P * pi + torch.square(psi) * g**(4 - 2 * i) * (1 - torch.square(rho)) * pi
	dem = pi**2 - torch.square(lambdak)
	alpha_next = alpha * num / dem
	return alpha_next

def get_lambdas(Dk_history, rhok_history, lambda1_history, lambda2_history, P, g, sigma1, sigma2, pi1, pi2, rhoz):
	Dk_kminus1 = Dk_history[:,-1]
	Dk_kminus2 = Dk_history[:,-2]
	psi_kminus1, psi_kminus2 = torch.sqrt(P/Dk_kminus1), torch.sqrt(P/Dk_kminus2)
	num1 = psi_kminus1 * psi_kminus2 * (g + torch.abs(rhok_history[:,-2])) * g * sgn(rhok_history[:,-1]) * sgn(rhok_history[:,-2]) * pi2 * sigma2 * (sigma2 - sigma1 * rhoz)
	num2 = psi_kminus1 * psi_kminus2 * (1 + g * torch.abs(rhok_history[:,-2])) * pi1 * sigma1 * (sigma1 - sigma2 * rhoz)
	dem1 = torch.sqrt(pi2**2 - torch.square(lambda2_history[:,-2])) * torch.sqrt(pi2**2 - torch.square(lambda2_history[:,-2]) - torch.square(psi_kminus2) * torch.square((g + torch.abs(rhok_history[:,-2]))) * pi2)
	dem2 = torch.sqrt(pi1**2 - torch.square(lambda1_history[:,-2])) * torch.sqrt(pi1**2 - torch.square(lambda1_history[:,-2]) - P * pi1 + torch.square(psi_kminus2) * g**2 * pi1 * (1 - torch.square(rhok_history[:,-2])))
	lambda1 = (num1 / dem1).view(Dk_history.shape[0], 1)
	lambda2 = (num2 / dem2).view(Dk_history.shape[0], 1)
	return lambda1, lambda2, psi_kminus1.view(Dk_history.shape[0], 1), psi_kminus2.view(Dk_history.shape[0], 1)


def errors_ber(y_true, y_pred):
    compare_result = torch.ne(y_true, y_pred).float()  
    res = torch.sum(compare_result)/(compare_result.shape[0] * compare_result.shape[1])  
    return res

args = get_args(False)
print('args = ', args.__dict__)

# message1, forward_noise1, feedback_noise1 = generate_data(args.K1, args.N, args.forward_SNR1, args.feedback_SNR1, args.num_samples)
# message2, forward_noise2, feedback_noise2 = generate_data(args.K2, args.N, args.forward_SNR2, args.feedback_SNR2, args.num_samples)

def ozarow(message1, forward_noise1, feedback_noise1, message2, forward_noise2, feedback_noise2, args):
	P = torch.tensor(args.power)
	g = args.g
	sigma1 = snr_db_2_sigma(args.forward_SNR1)
	sigma2 = snr_db_2_sigma(args.forward_SNR2)
	rhoz = args.rhoz
	pi1 = P + sigma1**2
	pi2 = P + sigma2**2
	num_samples = message1.shape[0]
	# store the past Dk and rhok
	Dk_history = torch.empty(num_samples, 0)
	rhok_history = torch.empty(num_samples, 0)
	lambda1_history = torch.empty(num_samples, 0)
	lambda2_history = torch.empty(num_samples, 0)

	Y1_history = torch.empty(num_samples, 0)
	Y2_history = torch.empty(num_samples, 0)

	# encoding 
	message1_index, theta1 = PAMmodulation(message1, args.K1)
	message2_index, theta2 = PAMmodulation(message2, args.K2)

	# first two transmission
	X1= torch.sqrt(P) * theta1
	X2= torch.sqrt(P) * theta2
	X_total = torch.cat([X1, X2], dim = 1)

	print(f"transmission: {1}, power: {torch.var(X1)}")
	print(f"transmission: {2}, power: {torch.var(X2)}")

	Y12 = Y11 = X1 + forward_noise1[:,0].view(num_samples, 1) # user1: ignore the second reception	
	Y22 = Y21 = X2 + forward_noise2[:,1].view(num_samples, 1) # user2: ignore the first reception

	Y1_history = torch.cat([Y1_history, Y11, Y12], dim = 1)
	Y2_history = torch.cat([Y2_history, Y21, Y22], dim = 1)

	for t in range(2, args.N + 1):
		# print("transmission", t)
		if t == 2:
			theta1_estimate = torch.sqrt(P) * Y12 /(P + sigma1**2)
			theta2_estimate = torch.sqrt(P) * Y22 /(P + sigma2**2)

			epsilon1 = theta1_estimate - theta1
			epsilon2 = theta2_estimate - theta2

			epsilon1_enc =  epsilon1 + feedback_noise1[:,0].view(X1.shape[0], 1)
			epsilon2_enc =  epsilon2 + feedback_noise2[:,1].view(X1.shape[0], 1)

			rho, lambda1, lambda2 = torch.zeros(num_samples, 1), torch.zeros(num_samples, 1), torch.zeros(num_samples, 1)
			
			Dk = 1 + g**2 + 2 * g * torch.abs(rho)
			
			alpha1_enc = torch.var(epsilon1_enc)
			alpha2_enc = torch.var(epsilon2_enc)

			rhok_history = torch.cat([rhok_history, rho, rho], dim = 1)# be careful of rho's dimension is 1 less
			Dk_history = torch.cat([Dk_history, Dk, Dk], dim = 1)
			lambda1_history = torch.cat([lambda1_history, lambda1, lambda1], dim = 1)
			lambda2_history = torch.cat([lambda2_history, lambda2, lambda2], dim = 1)

		else:
			# t transmission(t >= 3)
			Xt = torch.sqrt(P * 1.0/ Dk) * (epsilon1_enc/torch.sqrt(alpha1_enc)  + epsilon2_enc * g * sgn(rho)/torch.sqrt(alpha2_enc))
			Xt = normalize(Xt, P)
			print(f"transmission: {t}, power: {torch.var(Xt)}")
			X_total = torch.cat([X_total, Xt], dim = 1)
			Y1t = Xt + forward_noise1[:,t - 1].view(num_samples, 1)
			Y2t = Xt + forward_noise2[:,t - 1].view(num_samples, 1)

			
			lambda1, lambda2, psi_kminus1, psi_kminus2 = get_lambdas(Dk_history, rhok_history, lambda1_history, lambda2_history, P, g, sigma1, sigma2, pi1, pi2, rhoz)

			update_estimate1 = 	(psi_kminus1 * torch.sqrt(alpha1_enc) * (1 + g * torch.abs(rho))) * (pi1 * Y1t - lambda1 * Y1_history[:,-1].view(num_samples,1))/(pi1**2 - torch.square(lambda1)) 

			update_estimate2 = (psi_kminus1 * torch.sqrt(alpha2_enc) * (g + torch.abs(rho)) * sgn(rho)) * (pi2 * Y2t - lambda2 * Y2_history[:,-1].view(num_samples,1))/ (pi2**2 - torch.square(lambda1))
			
			
	
			theta1_estimate = theta1_estimate - update_estimate1
			theta2_estimate = theta2_estimate - update_estimate2
		
			# estimation error
			epsilon1 = theta1_estimate - theta1
			epsilon2 = theta2_estimate - theta2

			epsilon1_enc =  epsilon1 + feedback_noise1[:,t-1].view(X1.shape[0], 1)
			epsilon2_enc =  epsilon2 + feedback_noise2[:,t-1].view(X1.shape[0], 1)

			Y1_history = torch.cat([Y1_history, Y1t], dim = 1)
			Y2_history = torch.cat([Y2_history, Y2t], dim = 1)

			alpha1_enc, alpha2_enc, rho = update(alpha1_enc, alpha2_enc, rho, lambda1, lambda2, P, g, sigma1, sigma2, rhoz,  pi1, pi2, psi_kminus1)
			lambda1_history = torch.cat([lambda1_history, lambda1], dim = 1)
			lambda2_history = torch.cat([lambda2_history, lambda2], dim = 1)
			rhok_history = torch.cat([rhok_history, rho], dim = 1)
			Dk = 1 + g**2 + 2 * g * torch.abs(rho)
			Dk_history = torch.cat([Dk_history, Dk], dim = 1)


	# decoding 
	message1_index_pred, message1_pred = PAMdedulation(theta1_estimate, args.K1)
	message2_index_pred, message2_pred = PAMdedulation(theta2_estimate, args.K2)

	ber1 = errors_ber(message1, message1_pred)
	ber2 = errors_ber(message2, message2_pred)

	bler1 = 1 - sum(message1_index_pred == message1_index) / message1_index.shape[0]
	bler2 = 1 - sum(message2_index_pred == message2_index) / message2_index.shape[0]
	return message1_pred, message2_pred, ber1, ber2, bler1, bler2, X_total


message1, forward_noise1, feedback_noise1 = generate_data(args.K1, args.N, args.forward_SNR1, args.feedback_SNR1, args.num_samples)
message2, forward_noise2, feedback_noise2 = generate_data(args.K2, args.N, args.forward_SNR2, args.feedback_SNR2, args.num_samples)

message1_pred, message2_pred, ber1, ber2, bler1, bler2, X_total = ozarow(message1, forward_noise1, feedback_noise1, message2, forward_noise2, feedback_noise2, args)


print('the BER for user 1 is:', ber1)
print('the BER for user 2 is:', ber2)


print('the BLER for user 1 is:', bler1)
print('the BLER for user 2 is:', bler2)

print("total power", torch.var(X_total))


# nums = 100
# ber1_list = []
# ber2_list = []
# bler1_list = []
# bler2_list = []

# for i in range(nums):
# 	print(f"-----------------------{i}------------------------")
# 	message1, forward_noise1, feedback_noise1 = generate_data(args.K1, args.N, args.forward_SNR1, args.feedback_SNR1, args.num_samples)
# 	message2, forward_noise2, feedback_noise2 = generate_data(args.K2, args.N, args.forward_SNR2, args.feedback_SNR2, args.num_samples)

# 	message1_pred, message2_pred, ber1, ber2, bler1, bler2, X_total = ozarow(message1, forward_noise1, feedback_noise1, message2, forward_noise2, feedback_noise2, args)


# 	print('the BER for user 1 is:', ber1)
# 	print('the BER for user 2 is:', ber2)


# 	print('the BLER for user 1 is:', bler1)
# 	print('the BLER for user 2 is:', bler2)

# 	print("total power", torch.var(X_total))

# 	ber1_list.append(ber1)
# 	ber2_list.append(ber2)
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
