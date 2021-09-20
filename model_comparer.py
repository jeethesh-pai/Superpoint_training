import torch
from model_loader import SuperPointNet, SuperPointNet_gauss2


model, detector_1, detector_2 = SuperPointNet_gauss2(), SuperPointNet_gauss2(), SuperPointNet_gauss2()
pretrained = "../my_superpoint_pytorch/superPointNet_170000_checkpoint.pth.tar"
model.load_state_dict(torch.load(pretrained)['model_state_dict'])
detector1_model_weights = torch.load("colab_log/gauss_model_2.pt")
detector_1.load_state_dict(detector1_model_weights)
detector2_model_weights = torch.load("colab_log/joint_model_freeze.pt")
detector_2.load_state_dict(detector2_model_weights)
keys = model.state_dict().keys()
for key in keys:
    p_weight = model.state_dict()[key]
    d1_weight = detector_1.state_dict()[key].to('cpu')
    d2_weight = detector_2.state_dict()[key].to('cpu')
    diff1 = torch.subtract(p_weight, d1_weight).sum()
    diff2 = torch.subtract(p_weight, d2_weight).sum()
    print(f"{key}:{diff1}, {diff2}")

# superPointNet_gauss2_model = SuperPointNet_gauss2()
# model_dict = torch.load("colab_log/superPointNet_170000_checkpoint.pth.tar")
# superPointNet_gauss2_model.load_state_dict(model_dict['model_state_dict'])
# print(superPointNet_gauss2_model)



