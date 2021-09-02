import torch
from model_loader import SuperPointNetBatchNorm  # load the new model from module


# paths of the different weight files
old_weight_file_path = "superpoint_v1.pth"
new_weight_file_path = "new_superpoint_v1.pth"

old_state_dict = torch.load(old_weight_file_path)
new_model = SuperPointNetBatchNorm()
state_dict = new_model.state_dict()

# these layers are different for both modules. Either skip these or initialise to existing values
old_bias = ['conv1b.bias', 'conv2b.bias', 'conv3b.bias', 'conv4b.bias']
new_bias = ['bn1.bias', 'bn2.bias', 'bn3.bias', 'bn4.bias']
for key in state_dict:
    for old_key in old_state_dict:
        if old_key == key:
            state_dict[key] = old_state_dict[old_key]  # initializing similar layers with existing weights
for i in range(len(old_bias)):
    state_dict[new_bias[i]] = old_state_dict[old_bias[i]]  # initialising the BN bias with pre-trained Conv bias

torch.save(new_model.state_dict(), new_weight_file_path)  # saving the module
