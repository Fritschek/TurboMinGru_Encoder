import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


from numpy import arange
from numpy.random import mtrand
import math
import numpy as np
from mingru_stacks import StackedMambaMinGRU
from mingru_stacks import StackedMambaMinGRU_test
from mingru_stacks import WrappedMinGRU

##########################################
# ----- HELPER FUNCTIONS / CLASSES ------
##########################################

class TurboConfig:
    num_iteration = 6
    code_rate_k = 1
    dec_num_layer = 2
    dec_num_unit = 10
    dec_kernel_size = 5
    enc_num_unit = 100
    enc_num_layer = 3
    enc_kernel_size = 5
    num_iter_ft = 5
    #batch_size = 32
    block_len = 24
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
def circular_pad(tensor, pad):
    """Apply circular padding to a 1D tensor."""
    return torch.cat([tensor[:, :, -pad:], tensor, tensor[:, :, :pad]], dim=2)

class Interleaver(torch.nn.Module):
    """Handles both interleaving and de-interleaving based on a given permutation array."""
    
    def __init__(self, config):
        super(Interleaver, self).__init__()
        
        seed = np.random.randint(0, 1)
        rand_gen = mtrand.RandomState(seed)
        p_array = rand_gen.permutation(arange(config.block_len))
        self.set_parray(p_array)

    def set_parray(self, p_array):
        """Sets permutation array and its reverse."""
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

        reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            reverse_p_array[p_array[idx]] = idx
        self.reverse_p_array = torch.LongTensor(reverse_p_array).view(len(p_array))

    def _permute(self, inputs, permutation_array):
        """Permute the given input using the provided permutation array."""
        inputs = inputs.permute(1, 0, 2)
        res = inputs[permutation_array]
        return res.permute(1, 0, 2)

    def interleave(self, inputs):
        return self._permute(inputs, self.p_array)
    
    def deinterleave(self, inputs):
        return self._permute(inputs, self.reverse_p_array)


class ModuleLambda(nn.Module):
    def __init__(self, lambd):
        super(ModuleLambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
def build_encoder_block(num_layer, in_channels, out_channels, kernel_size, activation='elu'):
    layers = []
    layers.append(ModuleLambda(lambda x: torch.transpose(x, 1, 2)))
        
    for idx in range(num_layer):       
        # Add circular padding before the convolution, experimental
        # pad = kernel_size // 2
        # layers.append(ModuleLambda(lambda x: circular_pad(x, pad)))
        
        layers.append(nn.Conv1d(
            in_channels=in_channels if idx == 0 else out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding= kernel_size // 2, #0 if with circular padding, #kernel_size // 2  -> if without circular padding
            dilation=1,
            groups=1, 
            bias=True
        ))

        #layers.append(nn.LayerNorm([out_channels, 64])) # Experimental
        layers.append(ModuleLambda(lambda x: getattr(F, activation)(x)))
        #layers.append(nn.Dropout(0.1)) # Experimental
    
    layers.append(ModuleLambda(lambda x: torch.transpose(x, 1, 2)))
    return nn.Sequential(*layers)

class SaturatedSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Clamp the input to a specific range (e.g., [-1, 1])
        ctx.save_for_backward(input)
        return input.clamp(-1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradients only in the range [-1, 1], else 0
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < -1] = 0
        grad_input[input > 1] = 0
        return grad_input

##########################################
# ----- MinGRU CLASSES ------
##########################################
class ENC_GRUTurbo(nn.Module):
    def __init__(self, config, interleaver):
        super(ENC_GRUTurbo, self).__init__()
        self.config = config
        self.interleaver = interleaver
                
        # Initialize the minGRU layers
        self.enc_rnn_1 = StackedMambaMinGRU(config.enc_num_layer, 1, 7, bias=True)       
        self.enc_rnn_2 = StackedMambaMinGRU(config.enc_num_layer, 1, 7, bias=True)
        
        # Define activation function
        self.enc_act = F.elu  # Adjust as needed

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2).float()  # (batch_size, seq_len, 1)
    
        x_sys = self.enc_rnn_1(inputs)  # (batch_size, seq_len, hidden_size)

        # Second minGRU layer with interleaved inputs
        inputs_interleaved = self.interleaver.interleave(inputs)
        x_p1 = self.enc_rnn_2(inputs_interleaved)  # (batch_size, seq_len, hidden_size)

        # Concatenate outputs and apply power constraint
        x_tx = torch.cat([x_sys, x_p1], dim=2)  # (batch_size, seq_len, 2)
        codes = self.power_constraint(x_tx)  # Apply power constraint as defined

        return codes.squeeze(dim=2)  # (batch_size, seq_len, 2)

class ENC_GRUTurbo_h(nn.Module):
    def __init__(self, config, interleaver):
        super(ENC_GRUTurbo, self).__init__()
        self.config = config
        self.interleaver = interleaver
                
        # Initialize the minGRU layers
        self.enc_rnn_1 = StackedMambaMinGRU_test(config.enc_num_layer, 1, 3, bias=True)       
        self.enc_rnn_2 = StackedMambaMinGRU_test(config.enc_num_layer, 1, 3, bias=True)
        
        # Define activation function
        self.enc_act = F.elu  # Adjust as needed

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs, h_0_1=None, h_0_2=None):
        inputs = inputs.unsqueeze(dim=2).float()  # (batch_size, seq_len, 1)
    
        x_sys, h_0_1 = self.enc_rnn_1(inputs, h_0=h_0_1)  # (batch_size, seq_len, hidden_size)

        # Second minGRU layer with interleaved inputs
        inputs_interleaved = self.interleaver.interleave(inputs)
        x_p1, h_0_2 = self.enc_rnn_2(inputs_interleaved, h_0=h_0_2)  # (batch_size, seq_len, hidden_size)

        # Concatenate outputs and apply power constraint
        x_tx = torch.cat([x_sys, x_p1], dim=2)  # (batch_size, seq_len, 2)
        codes = self.power_constraint(x_tx)  # Apply power constraint as defined

        return codes.squeeze(dim=2), [h_0_1, h_0_2]  # (batch_size, seq_len, 2), hidden states
    
class DEC_GRUTurbo_test(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_GRUTurbo_test, self).__init__()
        
        self.config = config
                       
        self.interleaver = interleaver
        
        self.dec1_grus = torch.nn.ModuleList([
            StackedMambaMinGRU(config.dec_num_layer, 3, config.dec_num_unit)
            for _ in range(config.num_iteration)
        ])
        
        self.dec2_grus = torch.nn.ModuleList([
            StackedMambaMinGRU(config.dec_num_layer, 3, config.dec_num_unit)
            for _ in range(config.num_iteration)
        ])
        
        self.dec1_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, config.num_iter_ft) for _ in range(config.num_iteration)
        ])
        
        self.dec2_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft)
            for idx in range(config.num_iteration)
        ])
    
    def forward(self, received):
        bs = received.size(0)
        received = received.view(received.size(0), -1, 2)
        config = self.config
        received = received.to(next(self.parameters()).device).float()
    
        # Initial processing
        r_sys = received[:, :, 0].view((bs, config.block_len, 1))
        r_sys_int = self.interleaver.interleave(r_sys)
        r_par = received[:, :, 1].view((bs, config.block_len, 1))
        r_par_deint = self.interleaver.deinterleave(r_par)
    
        # Initialize prior
        prior = torch.zeros((bs, config.block_len, config.num_iter_ft), device=next(self.parameters()).device)
    
        # Turbo Decoder Loop
        for idx in range(config.num_iteration - 1):
            x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_grus[idx])
            x_plr_int = self.interleaver.interleave(x_plr - prior)
        
            x_dec, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_grus[idx])
            prior = self.interleaver.deinterleave(x_plr - x_plr_int)
        
        # Last round
        x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_grus[-1])
        x_plr_int = self.interleaver.interleave(x_plr - prior)
    
        x_dec, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_grus[-1])
        final = torch.sigmoid(self.interleaver.deinterleave(x_plr))
        return final.squeeze(dim=2)

    def _turbo_decoder_step(self, r_sys, r_par, prior, cnn):
        x_this_dec = torch.cat([r_sys, r_par, prior], dim=2)
        x_plr = cnn(x_this_dec)
        x_dec = x_plr
        #x_plr = linear(x_dec)
        return x_dec, x_plr
    
class DEC_GRUTurbo_test_low(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_GRUTurbo_test_low, self).__init__()
        
        self.config = config
                       
        self.interleaver = interleaver
        
        self.lin = torch.nn.Linear(3, 1)
        
        self.dec_rnn_1 = StackedMambaMinGRU(2, 3, 6, bias=True)       
        self.dec_rnn_2 = StackedMambaMinGRU(2, 3, 6, bias=True)
    
    def forward(self, received):
        bs = received.size(0)
        received = received.view(received.size(0), -1, 2)
        config = self.config
        received = received.to(next(self.parameters()).device).float()
    
        # Initial processing
        r_sys = received[:, :, 0].view((bs, config.block_len, 1))
        #r_sys_int = self.interleaver.interleave(r_sys)
        r_par = received[:, :, 1].view((bs, config.block_len, 1))
        prior = torch.zeros((bs, config.block_len, 1), device=next(self.parameters()).device)
        
        
        
        
        for idx in range(config.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par, prior], dim=2)
            x_sys_dec = self.dec_rnn_1(x_this_dec)
        
            x_sys_dec = self.interleaver.deinterleave(x_sys_dec)
        
            x_sys_dec = self.dec_rnn_2(x_sys_dec)
            
            prior = self.lin(x_sys_dec) - prior
            
        out = self.lin(x_sys_dec)
        
        out = out + r_sys
 
        final = torch.sigmoid(out)
        return final.squeeze(dim=2)
    
class DEC_GRUTurbo_m(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_GRUTurbo_m, self).__init__()
        self.config = config
        self.interleaver = interleaver
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.decoders_output1 = nn.ModuleList()
        self.decoders_output2 = nn.ModuleList()
        self.linea = nn.ModuleList()
        
        self.lin = nn.Linear(2, 1)
        
        self.decoders_rnn1 = torch.nn.ModuleList([
            StackedMambaMinGRU(config.dec_num_layer, 2, 6, bias=True)
            for _ in range(config.num_iteration)
            ])
        
        self.linea = nn.ModuleList([nn.Linear(2, 1) for _ in range(config.num_iteration)])

        self.decoders_rnn2 = torch.nn.ModuleList([
                StackedMambaMinGRU(config.dec_num_layer, 2, 6, bias=True)
                for _ in range(config.num_iteration)
            ])


    def set_parallel(self):
        for idx in range(self.config.num_iteration):
            self.decoders_rnn1[idx] = nn.DataParallel(self.decoders_rnn1[idx])
            self.decoders_rnn2[idx] = nn.DataParallel(self.decoders_rnn2[idx])
            self.decoders_output1[idx] = nn.DataParallel(self.decoders_output1[idx])
            self.decoders_output2[idx] = nn.DataParallel(self.decoders_output2[idx])

    def forward(self, received):
        bs = received.size(0)
        received = received.view(bs, -1, 2).type(torch.FloatTensor).to(self.device)

        # Initial processing of systematic and parity bits
        r_sys = received[:, :, 0].view(bs, self.config.block_len, 1)
        r_par = received[:, :, 1].view(bs, self.config.block_len, 1)
    
        # Interleaving and deinterleaving only once before the loop
        r_par_deint = self.interleaver.deinterleave(r_par)
        r_sys_int = self.interleaver.interleave(r_sys)
        
        # Initialize x_input as the concatenation of systematic and parity bits
        x_input = torch.cat([r_sys, r_par], dim=2)
        
        # Initialize extrinsic information to zero (for the first iteration)
        extrinsic_rnn1 = torch.zeros_like(r_sys).to(self.device)
        extrinsic_rnn2 = torch.zeros_like(r_sys).to(self.device)

        for idx in range(6):
            # Decode using the first RNN decoder (Decoder 1)
            inner = self.decoders_rnn1[idx](x_input)
            
            # Generate extrinsic information for the first decoder (extrinsic_rnn1)
            extrinsic_rnn1 = inner - r_sys  # Subtract intrinsic information
            
            # Interleave the output before passing to the second decoder (Decoder 2)
            extrinsic_rnn1 = self.interleaver.deinterleave(extrinsic_rnn1)
            
            # Decode using the second RNN decoder (Decoder 2)
            outer = self.decoders_rnn2[idx](extrinsic_rnn1)
            
            # Generate extrinsic information for the second decoder (extrinsic_rnn2)
            extrinsic_rnn2 = outer - self.interleaver.interleave(r_sys)  # Subtract intrinsic (interleaved systematic bits)
            
            # Interleave the extrinsic information from Decoder 2 before next iteration
            outer_int = self.linea[idx](self.interleaver.interleave(extrinsic_rnn2))
            
            # Update the input for the next iteration using extrinsic info
            x_input = torch.cat([r_sys, outer_int], dim=2)
            
            # Final output logits from the outer decoder
        logits = outer
    
        # Apply sigmoid activation to get final probabilities
        final = torch.sigmoid(self.lin(logits))
        
        return final.squeeze(dim=2)
        
        
    
class DEC_GRUTurbo(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_GRUTurbo, self).__init__()
        self.config = config
        self.interleaver = interleaver
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        input_size = 2 + self.config.num_iter_ft

        self.decoders_rnn1 = nn.ModuleList()
        self.decoders_output1 = nn.ModuleList()
        self.decoders_gate1 = nn.ModuleList()
        self.decoders_rnn2 = nn.ModuleList()
        self.decoders_output2 = nn.ModuleList()
        self.decoders_gate2 = nn.ModuleList()
        
        self.skip_linear = nn.ModuleList([nn.Linear(input_size, self.config.dec_num_unit) for _ in range(config.num_iteration)]) 


        for idx in range(self.config.num_iteration):
            # Output size for decoder 2
            if idx == self.config.num_iteration - 1:
                output_size_dec2 = 1
            else:
                output_size_dec2 = self.config.num_iter_ft
            output_size_dec1 = self.config.num_iter_ft

            # Decoder 1
            self.decoders_rnn1.append(
                StackedMambaMinGRU(config.dec_num_layer, input_size, self.config.dec_num_unit, bias=True)
            )
            self.decoders_output1.append(
                nn.Linear(self.config.dec_num_unit, output_size_dec1)
            )
            self.decoders_gate1.append(
                nn.Sequential(nn.Linear(self.config.dec_num_unit, output_size_dec1), nn.Sigmoid())
            )

            # Decoder 2
            self.decoders_rnn2.append(
                StackedMambaMinGRU(config.dec_num_layer, input_size, self.config.dec_num_unit, bias=True)
            )
            self.decoders_output2.append(
                nn.Linear(self.config.dec_num_unit, output_size_dec2)
            )
            self.decoders_gate2.append(
                nn.Sequential(nn.Linear(self.config.dec_num_unit, output_size_dec2), nn.Sigmoid())
            )

    def forward(self, received):
        bs = received.size(0)
        received = received.view(bs, -1, 2).type(torch.FloatTensor).to(self.device)

        # Initial processing of systematic and parity bits
        r_sys = received[:, :, 0].view(bs, self.config.block_len, 1)
        r_par = received[:, :, 1].view(bs, self.config.block_len, 1)
    
        # Interleaving and deinterleaving only once before the loop
        r_par_deint = self.interleaver.deinterleave(r_par)
        r_sys_int = self.interleaver.interleave(r_sys)

        # Initialize prior to zeros
        prior = torch.zeros((bs, self.config.block_len, self.config.num_iter_ft)).to(self.device)

        # Turbo Decoder Loop for all iterations except the last one
        for idx in range(self.config.num_iteration - 1):
            # Decoder 1
            x_dec1, x_plr1 = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.decoders_rnn1[idx], self.decoders_output1[idx], self.decoders_gate1[idx], idx)
            x_plr1_ex = x_plr1 - prior
            x_plr1_ex_int = self.interleaver.interleave(x_plr1_ex)

            # Decoder 2
            x_dec2, x_plr2 = self._turbo_decoder_step(r_sys_int, r_par, x_plr1_ex_int, self.decoders_rnn2[idx], self.decoders_output2[idx], self.decoders_gate2[idx], idx)
            x_plr2_ex = x_plr2 - x_plr1_ex_int
            prior = self.interleaver.deinterleave(x_plr2_ex)

        # Last round of decoding outside the loop
        x_dec1, x_plr1 = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.decoders_rnn1[-1], self.decoders_output1[-1], self.decoders_gate1[-1], idx)
        x_plr1_ex = x_plr1 - prior
        x_plr1_ex_int = self.interleaver.interleave(x_plr1_ex)

        x_dec2, x_plr2 = self._turbo_decoder_step(r_sys_int, r_par, x_plr1_ex_int, self.decoders_rnn2[-1], self.decoders_output2[-1], self.decoders_gate2[-1], idx)
        logits = self.interleaver.deinterleave(x_plr2)

        # Sigmoid activation and final output
        final = torch.sigmoid(logits)
        return final.squeeze(dim=2)

    def _turbo_decoder_step(self, r_sys, r_par, prior, rnn, linear, gate, idx):
        x_input = torch.cat([r_sys, r_par, prior], dim=2)
        x_dec = rnn(x_input)
        #Skip connection
        x_input_transformed = self.skip_linear[idx](x_input)
        x_dec = x_dec + x_input_transformed
        
        x_plr = linear(x_dec)
        g_out = gate(x_dec)  # Apply gating
        return x_dec, x_plr * g_out  # Element-wise multiplication with gate output

##########################################
# ----- TurboAE CNN CLASSES ------
##########################################

class ENC_CNNTurbo(nn.Module):
    def __init__(self, config, interleaver):
        super(ENC_CNNTurbo, self).__init__()
        
        self.config = config
        self.interleaver = interleaver
        
        self.enc_cnn_1 = build_encoder_block(config.enc_num_layer, config.code_rate_k, config.enc_num_unit, config.enc_kernel_size)
        self.enc_cnn_2 = build_encoder_block(config.enc_num_layer, config.code_rate_k, config.enc_num_unit, config.enc_kernel_size)
        
        self.enc_linear_1 = nn.Linear(config.enc_num_unit, 1)
        self.enc_linear_2 = nn.Linear(config.enc_num_unit, 1)

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2).float()
        #inputs = 2.0 * inputs - 1.0
        
        x_sys = self.enc_cnn_1(inputs)
        x_sys = F.elu(self.enc_linear_1(x_sys))
        
        x_p1 = self.enc_cnn_2(self.interleaver.interleave(inputs))
        x_p1 = F.elu(self.enc_linear_2(x_p1))

        x_tx = torch.cat([x_sys, x_p1], dim=2)
        codes = self.power_constraint(x_tx)

        return codes.squeeze(dim=2)



class DEC_CNNTurbo(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_CNNTurbo, self).__init__()
        
        self.config = config                      
        self.interleaver = interleaver
        
        self.dec1_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec2_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec1_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, config.num_iter_ft) for _ in range(config.num_iteration)
        ])
        
        self.dec2_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft)
            for idx in range(config.num_iteration)
        ])
    
    def forward(self, received):
        device = next(self.parameters()).device
        bs = received.size(0)
        received = received.view(received.size(0), -1, 2).to(device).float()
        config = self.config
    
        # Initial processing
        r_sys = received[:, :, 0].view((bs, config.block_len, 1))
        r_sys_int = self.interleaver.interleave(r_sys)
        r_par = received[:, :, 1].view((bs, config.block_len, 1))
        r_par_deint = self.interleaver.deinterleave(r_par)
    
        # Initialize prior
        prior = torch.zeros((bs, config.block_len, config.num_iter_ft), device=next(self.parameters()).device)
    
        # Turbo Decoder Loop
        for idx in range(config.num_iteration - 1):
            x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_cnns[idx], self.dec1_outputs[idx])
            x_plr_int = self.interleaver.interleave(x_plr - prior)
        
            x_dec, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_cnns[idx], self.dec2_outputs[idx])
            prior = self.interleaver.deinterleave(x_plr - x_plr_int)
        
        # Last round
        x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_cnns[-1], self.dec1_outputs[-1])
        x_plr_int = self.interleaver.interleave(x_plr - prior)
    
        x_dec, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_cnns[-1], self.dec2_outputs[-1])
        final = torch.sigmoid(self.interleaver.deinterleave(x_plr))
        return final.squeeze(dim=2)

    def _turbo_decoder_step(self, r_sys, r_par, prior, cnn, linear):
        x_this_dec = torch.cat([r_sys, r_par, prior], dim=2)
        x_dec = cnn(x_this_dec)
        x_plr = linear(x_dec)
        return x_dec, x_plr
    
class DEC_CNNTurbo_gated(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_CNNTurbo_gated, self).__init__()
        
        self.config = config
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.interleaver = interleaver
        
        self.dec1_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec2_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        # Output layers and corresponding gating layers
        self.dec1_outputs = torch.nn.ModuleList([
            nn.Linear(config.dec_num_unit, config.num_iter_ft) for _ in range(config.num_iteration)
        ])
        self.dec1_gates = torch.nn.ModuleList([
            nn.Sequential(nn.Linear(config.dec_num_unit, config.num_iter_ft), nn.Sigmoid()) for _ in range(config.num_iteration)
        ])
        self.dec2_outputs = torch.nn.ModuleList([
            nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft)
            for idx in range(config.num_iteration)
        ])
        self.dec2_gates = torch.nn.ModuleList([
            nn.Sequential(nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft), nn.Sigmoid())
            for idx in range(config.num_iteration)
        ])

    def forward(self, received):
        bs = received.size(0)
        received = received.view(bs, -1, 2).type(torch.FloatTensor).to(self.this_device)
        r_sys = received[:, :, 0].view((bs, self.config.block_len, 1))
        r_sys_int = self.interleaver.interleave(r_sys)
        r_par = received[:, :, 1].view((bs, self.config.block_len, 1))
        r_par_deint = self.interleaver.deinterleave(r_par)
        prior = torch.zeros((bs, self.config.block_len, self.config.num_iter_ft)).to(self.this_device)

        for idx in range(self.config.num_iteration - 1):
            x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_cnns[idx], self.dec1_outputs[idx], self.dec1_gates[idx])
            x_plr_int = self.interleaver.interleave(x_plr - prior)
            x_dec, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_cnns[idx], self.dec2_outputs[idx], self.dec2_gates[idx])
            prior = self.interleaver.deinterleave(x_plr - x_plr_int)

        x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_cnns[-1], self.dec1_outputs[-1], self.dec1_gates[-1])
        x_plr_int = self.interleaver.interleave(x_plr - prior)
        x_dec, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_cnns[-1], self.dec2_outputs[-1], self.dec2_gates[-1])
        final = torch.sigmoid(self.interleaver.deinterleave(x_plr))
        return final.squeeze(dim=2)

    def _turbo_decoder_step(self, r_sys, r_par, prior, cnn, linear, gate):
        x_this_dec = torch.cat([r_sys, r_par, prior], dim=2)
        x_dec = cnn(x_this_dec)
        x_plr = linear(x_dec)
        g_out = gate(x_dec)  # Apply gating
        return x_dec, x_plr * g_out  # Element-wise multiplication with gate output
    
##########################################
# ----- TurboAE Serial CNN CLASSES ------
##########################################


class ENC_CNNTurbo_serial(nn.Module):
    def __init__(self, config, interleaver):
        super(ENC_CNNTurbo_serial, self).__init__()
        
        self.config = config
        self.interleaver = interleaver
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Updated the encoder CNN layers. Now the first CNN produces k*Fc features and the second one processes them
        self.enc_cnn_1 = build_encoder_block(config.enc_num_layer, config.code_rate_k, config.code_rate_k * 10, config.enc_kernel_size)
        self.enc_cnn_2 = build_encoder_block(config.enc_num_layer, config.code_rate_k * 10, config.enc_num_unit, config.enc_kernel_size)
        
        self.enc_linear_1 = nn.Linear(config.code_rate_k * 10, 1)
        self.enc_linear_2 = nn.Linear(config.enc_num_unit, 1)

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2)
        inputs = 2.0 * inputs - 1.0
        
        # Output from first CNN structure
        x_sys = self.enc_cnn_1(inputs)
        x_sys_ = F.elu(self.enc_linear_1(x_sys))
        
        x_sys_ste = SaturatedSTE.apply(x_sys_)

        x_sys_interleaved = self.interleaver.interleave(x_sys_ste)       
        # Pass interleaved output through second CNN structure
        x_p1 = self.enc_cnn_2(x_sys_interleaved)
        
        x_p1 = F.elu(self.enc_linear_2(x_p1))

        x_tx = torch.cat([x_sys_, x_p1], dim=2)
        
        codes = self.power_constraint(x_tx)

        return codes.squeeze(dim=2)
    
    
    
class DEC_CNNTurbo_serial(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_CNNTurbo_serial, self).__init__()
        
        self.config = config
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.transform_layer = nn.Linear(config.dec_num_unit, 2)
        self.interleaver = interleaver
        
        self.dec1_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec2_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec1_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, config.num_iter_ft) for _ in range(config.num_iteration)
        ])
        
        self.dec2_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft)
            for idx in range(config.num_iteration)
        ])
    
    
    def forward(self, received):
        bs = received.size(0)
        received = received.view(received.size(0), -1, 2)
        received = received.type(torch.FloatTensor).to(self.this_device)
        
        # Initialize prior
        prior = torch.zeros((bs, self.config.block_len, self.config.num_iter_ft)).to(self.this_device)
        
        # Iterative Decoding Loop
        for idx in range(self.config.num_iteration-1):
            x_dec1, x_plr1 = self._turbo_decoder_step(received, prior, self.dec1_cnns[idx], self.dec1_outputs[idx])
            x_plr1_extrinsic = x_plr1 - prior  # Compute extrinsic information
            x_plr1_extrinsic_int = self.interleaver.deinterleave(x_plr1_extrinsic)  # Interleave the extrinsic information
            
            x_dec1_transf = self.transform_layer(x_dec1)
            x_dec2, x_plr2 = self._turbo_decoder_step(x_dec1_transf, x_plr1_extrinsic_int, self.dec2_cnns[idx], self.dec2_outputs[idx])
            x_plr2_extrinsic = x_plr2 - x_plr1_extrinsic_int  # Compute extrinsic information
            prior = self.interleaver.interleave(x_plr2_extrinsic)  # Deinterleave the extrinsic information for the next iteration
        # -----------    
        # last round
        x_dec1, x_plr1 = self._turbo_decoder_step(received, prior, self.dec1_cnns[-1], self.dec1_outputs[-1])
        x_plr1_extrinsic = x_plr1 - prior  # Compute extrinsic information
        x_plr1_extrinsic_int = self.interleaver.deinterleave(x_plr1_extrinsic)  # Interleave the extrinsic information
            
        # Decoder 2 Processing
        x_dec1_transf = self.transform_layer(x_dec1)
        x_dec2, x_plr2 = self._turbo_decoder_step(x_dec1_transf, x_plr1_extrinsic_int, self.dec2_cnns[-1], self.dec2_outputs[-1])
        out = torch.sigmoid(self.interleaver.interleave(x_plr2))
        
        return out.squeeze(dim=2)  # Return the final output of Decoder 2

    def _turbo_decoder_step(self, input, prior, cnn, linear):
        x_this_dec = torch.cat([input, prior], dim=2)
        x_dec = cnn(x_this_dec)
        x_plr = linear(x_dec)
        return x_dec, x_plr
    
##########################################
# ----- TurboAE GRU CLASSES ------
##########################################

class ENC_rnn_rate2(nn.Module):
    def __init__(self, config, interleaver):
        super(ENC_rnn_rate2, self).__init__()
        self.config = config
        self.interleaver = interleaver
        self.enc_rnn_1       = torch.nn.GRU(1, config.enc_num_unit,
                                           num_layers=config.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_1    = torch.nn.Linear(2*config.enc_num_unit, 1)

        self.enc_rnn_2       = torch.nn.GRU(1, config.enc_num_unit,
                                           num_layers=config.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_2    = torch.nn.Linear(2*config.enc_num_unit, 1)


    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2).float()
        x_sys, _   = self.enc_rnn_1(inputs)
        x_sys      = F.elu(self.enc_linear_1(x_sys))

        x_sys_int  = self.interleaver.interleave(inputs)

        x_p2, _    = self.enc_rnn_2(x_sys_int)
        x_p2       = F.elu(self.enc_linear_2(x_p2))

        x_tx       = torch.cat([x_sys, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)
        return codes



class DEC_LargeMinRNNTurbo(nn.Module):
    def __init__(self, config, interleaver, dec_rnn='gru', dropout=0.0, activation='linear'):
        super(DEC_LargeMinRNNTurbo, self).__init__()
        
        self.config = config
        self.interleaver = interleaver  # Using your CNN-based interleaver
        self.dropout = nn.Dropout(dropout)
        self.activation_type = activation
        
        # Choose RNN model based on the dec_rnn argument
        if dec_rnn == 'gru':
            RNN_MODEL = nn.GRU
        elif dec_rnn == 'lstm':
            RNN_MODEL = nn.LSTM
        else:
            RNN_MODEL = nn.RNN

        # Initialize RNN layers for decoders
        self.dec1_rnns = nn.ModuleList([
            WrappedMinGRU(2 + config.num_iter_ft, config.dec_num_unit, num_layers=2, 
                      bidirectional=True) 
            for _ in range(config.num_iteration)
        ])

        self.dec2_rnns = nn.ModuleList([
            WrappedMinGRU(2 + config.num_iter_ft, config.dec_num_unit, num_layers=2, 
                      bidirectional=True) 
            for _ in range(config.num_iteration)
        ])
        
        # Initialize output layers for each decoding step
        self.dec1_outputs = nn.ModuleList([
            nn.Linear(2 * config.dec_num_unit, config.num_iter_ft) 
            for _ in range(config.num_iteration)
        ])
        
        self.dec2_outputs = nn.ModuleList([
            nn.Linear(2 * config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft) 
            for idx in range(config.num_iteration)
        ])
    
    def forward(self, received):
        device = next(self.parameters()).device
        bs = received.size(0)
        received = received.view(bs, self.config.block_len, 2).to(device).float()
        
        # Initial processing of received signals
        r_sys = received[:, :, 0].unsqueeze(-1)
        r_par1 = received[:, :, 1].unsqueeze(-1)
        
        # Initialize prior
        prior = torch.zeros((bs, self.config.block_len, self.config.num_iter_ft), device=device)
        
        # Turbo Decoder Loop
        for idx in range(self.config.num_iteration - 1):
            # First Decoder Step
            x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par1, prior, self.dec1_rnns[idx], self.dec1_outputs[idx])
            x_plr_int = self.interleaver.interleave(x_plr - prior)
            
            # Second Decoder Step
            x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par1, x_plr_int, self.dec2_rnns[idx], self.dec2_outputs[idx])
            prior = self.interleaver.deinterleave(x_plr - x_plr_int)
        
        # Last round without prior update
        x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par1, prior, self.dec1_rnns[-1], self.dec1_outputs[-1])
        x_plr_int = self.interleaver.interleave(x_plr - prior)
        
        x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par1, x_plr_int, self.dec2_rnns[-1], self.dec2_outputs[-1])
        final = torch.sigmoid(self.interleaver.deinterleave(x_plr))
        
        return final.squeeze(dim=2)

    def _turbo_decoder_step(self, r_sys, r_par, prior, rnn_layer, linear_layer):
        # Concatenate systematic, parity, and prior information
        x_this_dec = torch.cat([r_sys, r_par, prior], dim=2)
        
        # Pass through RNN
        x_dec, _ = rnn_layer(x_this_dec) 
        
        # Apply linear layer, dropout, and activation function
        x_plr = self.dropout(linear_layer(x_dec))
        
        return x_dec, x_plr

class DEC_LargeRNNTurbo(nn.Module):
    def __init__(self, config, interleaver, dec_rnn='gru', dropout=0.0, activation='linear'):
        super(DEC_LargeRNNTurbo, self).__init__()
        
        self.config = config
        self.interleaver = interleaver  # Using your CNN-based interleaver
        self.dropout = nn.Dropout(dropout)
        self.activation_type = activation
        
        # Choose RNN model based on the dec_rnn argument
        if dec_rnn == 'gru':
            RNN_MODEL = nn.GRU
        elif dec_rnn == 'lstm':
            RNN_MODEL = nn.LSTM
        else:
            RNN_MODEL = nn.RNN

        # Initialize RNN layers for decoders
        self.dec1_rnns = nn.ModuleList([
            RNN_MODEL(2 + config.num_iter_ft, config.dec_num_unit, num_layers=2, 
                      batch_first=True, bidirectional=True, dropout=dropout) 
            for _ in range(config.num_iteration)
        ])

        self.dec2_rnns = nn.ModuleList([
            RNN_MODEL(2 + config.num_iter_ft, config.dec_num_unit, num_layers=2, 
                      batch_first=True, bidirectional=True, dropout=dropout) 
            for _ in range(config.num_iteration)
        ])
        
        # Initialize output layers for each decoding step
        self.dec1_outputs = nn.ModuleList([
            nn.Linear(2 * config.dec_num_unit, config.num_iter_ft) 
            for _ in range(config.num_iteration)
        ])
        
        self.dec2_outputs = nn.ModuleList([
            nn.Linear(2 * config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft) 
            for idx in range(config.num_iteration)
        ])
    
    def forward(self, received):
        device = next(self.parameters()).device
        bs = received.size(0)
        received = received.view(bs, self.config.block_len, 2).to(device).float()
        
        # Initial processing of received signals
        r_sys = received[:, :, 0].unsqueeze(-1)
        r_par1 = received[:, :, 1].unsqueeze(-1)
        
        # Initialize prior
        prior = torch.zeros((bs, self.config.block_len, self.config.num_iter_ft), device=device)
        
        # Turbo Decoder Loop
        for idx in range(self.config.num_iteration - 1):
            # First Decoder Step
            x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par1, prior, self.dec1_rnns[idx], self.dec1_outputs[idx])
            x_plr_int = self.interleaver.interleave(x_plr - prior)
            
            # Second Decoder Step
            x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par1, x_plr_int, self.dec2_rnns[idx], self.dec2_outputs[idx])
            prior = self.interleaver.deinterleave(x_plr - x_plr_int)
        
        # Last round without prior update
        x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par1, prior, self.dec1_rnns[-1], self.dec1_outputs[-1])
        x_plr_int = self.interleaver.interleave(x_plr - prior)
        
        x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par1, x_plr_int, self.dec2_rnns[-1], self.dec2_outputs[-1])
        final = torch.sigmoid(self.interleaver.deinterleave(x_plr))
        
        return final.squeeze(dim=2)

    def _turbo_decoder_step(self, r_sys, r_par, prior, rnn_layer, linear_layer):
        # Concatenate systematic, parity, and prior information
        x_this_dec = torch.cat([r_sys, r_par, prior], dim=2)
        
        # Pass through RNN
        x_dec, _ = rnn_layer(x_this_dec) 
        
        # Apply linear layer, dropout, and activation function
        x_plr = self.dropout(linear_layer(x_dec))
        x_plr = self._apply_activation(self.activation_type, x_plr)
        
        return x_dec, x_plr

    def _apply_activation(self, act_type, inputs):
        if act_type == 'tanh':
            return torch.tanh(inputs)
        elif act_type == 'relu':
            return F.relu(inputs)
        elif act_type == 'sigmoid':
            return torch.sigmoid(inputs)
        elif act_type == 'linear':
            return inputs
        return inputs
    

##########################################
# ----- Interleaver Experiments ------
##########################################

class InterleaveFunction(Function):
    @staticmethod
    def forward(ctx, x, permutation):
        ctx.permutation = permutation
        x = x.permute(1, 0, 2)
        res = x[permutation]
        return res.permute(1, 0, 2)

    @staticmethod
    def backward(ctx, grad_output):
        inverse_permutation = torch.argsort(ctx.permutation)
        grad_output = grad_output.permute(1, 0, 2)
        grad_input = grad_output[inverse_permutation]
        return grad_input.permute(1, 0, 2), None

class LearnableInterleaver(Interleaver):
    def __init__(self, config):
        super().__init__(config)
        
    def _permute(self, inputs, permutation_array):
        return InterleaveFunction.apply(inputs, permutation_array)
    
#### This tries to create a soft permutation, which is also learnable
    
class GumbelInterleaveFunction(Function):
    @staticmethod
    def forward(ctx, x, logits, temperature=1.001):
        # Sample from Gumbel(0, 1)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        
        # Apply Gumbel Softmax trick
        y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)        
        # Store for backward pass
        ctx.save_for_backward(y)
        
        return GumbelInterleaver._apply_permutation(x, y)
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        
        grad_input = GumbelInterleaver._apply_permutation(grad_output, y)
        return grad_input, None, None

class GumbelInterleaver(nn.Module):
    def __init__(self, config, permutation=None):
        super(GumbelInterleaver, self).__init__()
        input_dim = config.block_len
        self.logits = nn.Parameter(torch.randn(input_dim, input_dim))
        
        if permutation is not None:
            self.logits.data = self.initialize_logits_from_permutation(permutation)
    
    def interleave(self, x, temperature=1.001):
        return GumbelInterleaveFunction.apply(x, self.logits, temperature)
    
    def deinterleave(self, x, temperature=1.001):
        # Use transpose of the softmax matrix for de-interleaving
        y_transpose = F.softmax(self.logits / temperature, dim=-1).t()
        return self._apply_permutation(x, y_transpose)

    @staticmethod
    def _apply_permutation(x, y):
        """Helper function to apply permutation based on y."""
        original_shape = x.shape
        x_2d = x.reshape(original_shape[0] * original_shape[2], original_shape[1])
        out_2d = torch.mm(x_2d, y.t())
        return out_2d.reshape(original_shape)
    
    @staticmethod
    def initialize_logits_from_permutation(permutation):
        """
        Initialize the logits matrix from a given permutation.
        """
        block_len = len(permutation)
        logits = -1e6 * torch.ones(block_len, block_len)  # Using a large negative value to make other entries negligible
        for idx, perm_idx in enumerate(permutation):
            logits[idx, perm_idx] = 1e6  # Setting a large positive value to the desired permutation position
        return F.softmax(logits, dim=-1)  # Convert to softmaxed version
    

##########################################
# ----- 2D CNN Experiments ------
##########################################



def circular_pad_2d(x, pad):
    pad_h, pad_w = pad
    # Circular padding for the height (sequence length)
    top_pad = x[:, :, -pad_h:, :]
    bottom_pad = x[:, :, :pad_h, :]
    x = torch.cat([top_pad, x, bottom_pad], dim=2)
    
    return x


def build_encoder_block_2d(num_layer, in_channels, out_channels, kernel_size, activation='elu'):
    layers = []
    
    def print_shape(x):
        print(x.shape)
        return x
    
    for idx in range(num_layer):
        in_ch = in_channels if idx == 0 else out_channels
        
        # Add modified padding before the convolution for 2D
        pad_h = kernel_size // 2
        pad_w = 1  # Since the width is 2, we can pad by 1 on each side
        layers.append(ModuleLambda(print_shape))
        layers.append(ModuleLambda(lambda x, pad_h=pad_h, pad_w=pad_w: circular_pad_2d(x, (pad_h, pad_w))))
        layers.append(ModuleLambda(print_shape))
        
        layers.append(nn.Conv2d(
            in_channels=in_ch, 
            out_channels=out_channels, 
            kernel_size=(kernel_size, 3),  # Adjusted kernel size to consider both sequences as spatial dimensions
            stride=1, 
            padding=(0, 1),  # We handle the width padding using our modified padding function
            dilation=1, 
            groups=1, 
            bias=True
        ))

        #norm_layer = nn.LayerNorm([out_channels, 64, 2])  # We set the height (sequence length) to a default value; it will adjust based on the input tensor
        #layers.append(ModuleLambda(lambda x, norm_layer=norm_layer: norm_layer(x)))
        layers.append(ModuleLambda(lambda x: getattr(F, activation)(x)))
        layers.append(nn.Dropout(0.1))
    
    return nn.Sequential(*layers)


class ENC_CNNTurbo_2D(nn.Module):
    def __init__(self, config, interleaver):
        super(ENC_CNNTurbo_2D, self).__init__()
        
        self.config = config
        self.interleaver = interleaver
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Encoder CNN and Linear layers for the combined 2D input
        self.enc_cnn = build_encoder_block_2d(config.enc_num_layer, 1, self.config.enc_num_unit, config.enc_kernel_size)  # Setting in_channels to 1
        self.enc_linear = nn.Linear(config.enc_num_unit, 1)

    def set_parallel(self):
        self.enc_cnn = nn.DataParallel(self.enc_cnn)
        self.enc_linear = nn.DataParallel(self.enc_linear)

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2)
        inputs = 2.0 * inputs - 1.0
        
        x_sys = inputs
        x_p1 = self.interleaver.interleave(inputs)
        
        # Stack the sequences along the last dimension and reshape for 2D CNN
        x_combined = torch.stack([x_sys.squeeze(-1), x_p1.squeeze(-1)], dim=-1).unsqueeze(1)
        print("before cnn",x_combined.shape)
        x_combined = self.enc_cnn(x_combined)
        print("after cnn",x_combined.shape)
        
        x_tx = F.elu(self.enc_linear(x_combined))

        codes = self.power_constraint(x_tx)

        return codes.squeeze(dim=2)
    
class STEBinarize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        # Save inputs for backward computation
        ctx.save_for_backward(inputs)
        
        # Binarize the inputs
        outputs = torch.sign(inputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Zero out gradients where the absolute value of the input is greater than 1
        grad_output[input > 1.0] = 0
        grad_output[input < -1.0] = 0
        # Clamp the gradient values
        grad_output = torch.clamp(grad_output, -0.25, +0.25)
        
        return grad_output