import cv2
import torch
import yaml
from Data_loader import TLSScanData
from model_loader import SuperPointNet, load_model, SuperPointNetBatchNorm, SuperPointNetBatchNorm2
import torch.optim as optim
from utils import detector_loss, descriptor_loss_2, descriptor_loss_3, descriptor_loss_modified
from torchsummary import summary
import copy
from collections import namedtuple
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import tqdm
import warnings
import argparse
from torchviz import make_dot

warnings.simplefilter("ignore")


class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues. This wraps model with dictionary outputs
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor):
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            if p.grad is not None:
                # print(n[:-6], p.grad.abs().mean())
                layers.append(n[:-6])
                ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


parser = argparse.ArgumentParser(description="This scripts helps to train Superpoint for detector training after"
                                             "Homographic adaptation labels are made as mentioned in the paper")
parser.add_argument('--config', help='Path to config file',
                    default="joint_training.yaml")
args = parser.parse_args()

config_file_path = args.config
with open(config_file_path) as path:
    config = yaml.full_load(path)
batch_size = config['model']['batch_size']
numHomIter = config['data']['augmentation']['homographic']['num']
det_threshold = config['model']['detection_threshold']  # detection threshold to threshold the detector heatmap
size = config['data']['preprocessing']['resize']  # width, height
train_set = TLSScanData(transform=None, task='train', **config)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, prefetch_factor=2)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True,
#                                            prefetch_factor=4, num_workers=2)
val_set = TLSScanData(transform=None, task='validation', **config)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['model']['eval_batch_size'], shuffle=False,
                                         prefetch_factor=2)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['model']['eval_batch_size'], shuffle=False,
#                                          pin_memory=True, prefetch_factor=4, num_workers=2)
Net = SuperPointNetBatchNorm2()
optimizer = optim.Adam(Net.parameters(), lr=config['model']['learning_rate'])
epochs = 0
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_weights = load_model(config['pretrained'], Net)
Net.load_state_dict(model_weights)
Net.to(device)
summary(Net, (1, size[1], size[0]), batch_size=1)
wrappedModel = ModelWrapper(Net)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
if config['data']['detector_training']:  # we bootstrap the Superpoint detector using homographic adapted labels
    writer = SummaryWriter(log_dir="../logs/homoAdaptIter1")
    writer.add_graph(wrappedModel, input_to_model=torch.ones(size=(2, 1, size[1], size[0])).to(device))
    max_iter = config['train_iter']  # also called as epochs
    n_iter = 0
    prev_val_loss = None
    Net.train(mode=True)
    old_state_dict = {}
    new_state_dict = {}
    for key in Net.state_dict():
        old_state_dict[key] = Net.state_dict()[key].clone()
    while n_iter < max_iter:  # epochs can be lesser since no random homographic adaptation is involved
        running_loss = 0
        train_bar = tqdm.tqdm(train_loader)
        for i, sample in enumerate(train_bar):  # make sure the homographic adaptation is false here
            # plt.imshow(sample['label'][0, :, :].numpy().squeeze(), cmap='gray')
            # plt.show()
            # plt.imshow(sample['label'][1, :, :].numpy().squeeze(), cmap='gray')
            # plt.show()
            sample['image'] = sample['image'].to(device)
            sample['label'] = sample['label'].to(device)
            optimizer.zero_grad()
            out = Net(sample['image'])
            semi, _ = out['semi'], out['desc']
            det_out = detector_loss(sample['label'], semi, device=device)
            loss = det_out['loss']
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_description(f"Training Epoch -- {n_iter + 1} / {max_iter} - Loss: {running_loss / (i + 1)}")
        running_val_loss = 0
        val_bar = tqdm.tqdm(val_loader)
        Net.train(mode=False)
        for j, val_sample in enumerate(val_bar):
            val_sample['image'] = val_sample['image'].to(device)
            val_sample['label'] = val_sample['label'].to(device)
            with torch.no_grad():
                val_out = Net(val_sample['image'])
                val_det_out = detector_loss(val_sample['label'], val_out['semi'], device=device)
                running_val_loss += val_det_out['loss'].item()
            val_bar.set_description(f"Validation -- Epoch- {n_iter + 1} / {max_iter} - Validation loss: "
                                    f"{running_val_loss / (j + 1)}")
        running_val_loss /= len(val_loader)
        if prev_val_loss is None or prev_val_loss > running_val_loss:
            prev_val_loss = running_val_loss
            print('saving best model .... ')
            torch.save(copy.deepcopy(Net.state_dict()), "../homoAdaptiter1_BN.pt")
        scheduler.step(prev_val_loss)
        writer.add_scalar('Loss', running_loss, n_iter + 1)
        writer.add_scalar('Val_loss', running_val_loss, n_iter + 1)
        writer.flush()
        n_iter += 1
else:  # start descriptor training with the homographically adapted model
    writer = SummaryWriter(log_dir="logs/descriptor_training")  # /content/drive/MyDrive/logs/joint_training
    writer.add_graph(wrappedModel, input_to_model=torch.ones(size=(2, 1, size[1], size[0])).to(device))
    max_iter = config['train_iter']  # also called as epochs
    det_threshold = config['model']['detection_threshold']
    n_iter = 0
    prev_val_loss = None
    old_state_dict = {}
    new_state_dict = {}
    for key in Net.state_dict():
        old_state_dict[key] = Net.state_dict()[key].clone()
    # tot_params = sum(1 for _ in Net.parameters())
    # count = 0
    # for params in Net.parameters():
    #     if count < 40:
    #         params.requires_grad = False
    #         count += 1
    # for params in Net.named_parameters():
    #     if params[1].requires_grad is False:
    #         print(params[0])
    count_nan = 0
    while n_iter < max_iter:  # epochs can be lesser since no random homographic adaptation is involved
        running_loss = 0
        optimizer.zero_grad()
        train_bar = tqdm.tqdm(train_loader)
        Net.train(mode=True)
        for i, sample in enumerate(train_bar):  # make sure the homographic adaptation is set to true here
            # fig, axes = plt.subplots(2, 2)
            # axes[0, 0].imshow(sample['image'].numpy()[0, 0, :, :].squeeze(), cmap='gray')
            # axes[1, 0].imshow(sample['warped_image'][0, 0, :, :].numpy().squeeze(), cmap='gray')
            # axes[0, 1].imshow(sample['label'][0, 0, :, :].numpy().squeeze(), cmap='gray')
            # axes[1, 1].imshow(sample['warped_label'][0, 0, :, :].numpy().squeeze(), cmap='gray')
            # fig, axes = plt.subplots(1, 2)
            # axes[0].imshow(sample['image'].numpy().squeeze(), cmap='gray')
            # axes[1].imshow(cv2.warpPerspective(sample['image'].numpy().squeeze(), sample['homography'].numpy().squeeze(),
            #                                    flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR, dsize=(320, 216)), cmap='gray')
            # plt.show()
            sample['image'] = sample['image'].to(device)
            sample['label'] = sample['label'].to(device)
            sample['warped_image'] = sample['warped_image'].to(device)
            sample['warped_label'] = sample['warped_label'].to(device)
            if torch.sum(torch.isnan(sample['warped_image'])):
                print('\ncaught nan in warped image', count_nan, 'times')
                count_nan += 1
                continue
            out = Net(sample['image'])
            out_warp = Net(sample['warped_image'])
            semi, desc = out['semi'], out['desc']
            semi_warped, desc_warp = out_warp['semi'], out_warp['desc']
            det_loss = detector_loss(sample['label'], semi, device=device)
            det_warp_loss = detector_loss(sample['warped_label'], semi_warped, device=device)
            desc_loss = descriptor_loss_modified(desc, desc_warp, homography=sample['inv_homography'],
                                                 margin_neg=config['model']['negative_margin'],
                                                 margin_pos=config['model']['positive_margin'],
                                                 lambda_d=config['model']['lambda_d'],
                                                 threshold=config['model']['descriptor_dist'],
                                                 valid_mask=sample['warped_mask'])
            total_loss = det_loss['loss'] + det_warp_loss['loss'] + config['model']['lambda_loss'] * desc_loss
            total_loss.backward()
            running_loss += total_loss.item()
            if (i + 1) * batch_size % config['model']['update_batch_size'] == 0:  # performing big batch updates
                optimizer.step()
                optimizer.zero_grad()
            train_bar.set_description(f"Training Epoch -- {n_iter + 1} / {max_iter} - Loss: {running_loss / (i + 1)}")
        running_val_loss = 0
        val_bar = tqdm.tqdm(val_loader)
        Net.train(mode=False)
        for j, val_sample in enumerate(val_bar):
            val_sample['image'] = val_sample['image'].to(device)
            val_sample['label'] = val_sample['label'].to(device)
            val_sample['warped_image'] = val_sample['warped_image'].to(device)
            val_sample['warped_label'] = val_sample['warped_label'].to(device)
            if torch.sum(torch.isnan(val_sample['warped_image'])):
                print('\ncaught nan in warped image', count_nan, 'times')
                count_nan += 1
                continue
            with torch.no_grad():
                out = Net(val_sample['image'])
                out_warp = Net(val_sample['warped_image'])
                semi, desc = out['semi'], out['desc']
                semi_warped, desc_warp = out_warp['semi'], out_warp['desc']
                det_loss = detector_loss(val_sample['label'], semi, device)
                det_warp_loss = detector_loss(val_sample['label'], semi_warped, device)
                desc_loss = descriptor_loss_2(desc, desc_warp, homography=val_sample['inv_homography'],
                                              margin_neg=config['model']['negative_margin'],
                                              margin_pos=config['model']['positive_margin'],
                                              lambda_d=config['model']['lambda_d'],
                                              threshold=config['model']['descriptor_dist'],
                                              valid_mask=val_sample['warped_mask'])
                total_loss = det_loss['loss'] + det_warp_loss['loss'] + config['model']['lambda_loss'] * desc_loss
                running_val_loss += total_loss.item()
                val_bar.set_description(f"Validation -- Epoch- {n_iter + 1} / {max_iter} - Validation loss: "
                                        f"{running_val_loss / (j + 1)}")
        running_val_loss /= len(val_loader)
        if prev_val_loss is None or prev_val_loss > running_val_loss:
            prev_val_loss = running_val_loss
            print('saving best model .... ')
            torch.save(copy.deepcopy(Net.state_dict()), f"../descriptorTrainingAfterIter2myloss_{n_iter+1}.pt")
        scheduler.step(prev_val_loss)
        writer.add_scalar('Loss', running_loss, n_iter + 1)
        writer.add_scalar('Val_loss', running_val_loss, n_iter + 1)
        writer.flush()
        n_iter += 1
    writer.close()
