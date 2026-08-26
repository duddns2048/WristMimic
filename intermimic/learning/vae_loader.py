from utils import torch_utils
import torch
import torch.nn as nn
from easydict import EasyDict as edict


def load_mlp(loading_keys, checkpoint, actvation_func):
    
    loading_keys_linear = [k for k in loading_keys if k.endswith('weight')]
    nn_modules = []
    for idx, key in enumerate(loading_keys_linear):
        if len(checkpoint['model'][key].shape) == 1: # layernorm
            layer = torch.nn.LayerNorm(*checkpoint['model'][key].shape[::-1])
            nn_modules.append(layer)
        elif len(checkpoint['model'][key].shape) == 2: # nn
            layer = nn.Linear(*checkpoint['model'][key].shape[::-1])
            nn_modules.append(layer)
            if idx < len(loading_keys_linear) - 1:
                nn_modules.append(actvation_func())
        else:
            raise NotImplementedError
        
    net = nn.Sequential(*nn_modules)
    
    state_dict = net.state_dict()
    
    for idx, key_affix in enumerate(state_dict.keys()):
        state_dict[key_affix].copy_(checkpoint['model'][loading_keys[idx]])
        
    for param in net.parameters():
        param.requires_grad = False
        
    return net

def load_z_decoder(checkpoint, activation = "silu", z_type = "vae", device = "cpu"):
    actvation_func = torch_utils.activation_facotry(activation)
    key_name = "a2c_network.actor_mlp"
    loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(key_name)] + ["a2c_network.mu.weight", 'a2c_network.mu.bias']
        
    actor = load_mlp(loading_keys, checkpoint, actvation_func)
    
    actor.to(device)
    actor.eval()
    net_dict = edict()
    
    net_dict.decoder = actor
        
    prior_loading_keys = [k for k in checkpoint['model'].keys() if k.startswith("a2c_network.z_prior.")]
    z_prior = load_mlp(prior_loading_keys, checkpoint, actvation_func)
    z_prior.append(actvation_func())
    z_prior_mu = load_linear('a2c_network.z_prior_mu', checkpoint=checkpoint)
    
    z_prior.eval()
    z_prior_mu.eval()
    net_dict.z_prior = z_prior.to(device)
    net_dict.z_prior_mu = z_prior_mu.to(device)

    z_prior_logvar = load_linear('a2c_network.z_prior_logvar', checkpoint=checkpoint)
    z_prior_logvar.eval()
    net_dict.z_prior_logvar = z_prior_logvar.to(device)
    return net_dict


def load_linear(net_name, checkpoint):
    net = nn.Linear(checkpoint['model'][net_name + '.weight'].shape[1], checkpoint['model'][net_name + '.weight'].shape[0])
    state_dict = net.state_dict()
    state_dict['weight'].copy_(checkpoint['model'][net_name + '.weight'])
    state_dict['bias'].copy_(checkpoint['model'][net_name + '.bias'])
    
    return net

