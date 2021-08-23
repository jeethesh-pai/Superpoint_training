import torch
from utils import detector_loss, descriptor_loss_2


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. From Daniel De Tone Implementation"""

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x: torch.Tensor) -> dict:
        """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return {'semi': semi, 'desc': desc}  # semi is the detector head and desc is the descriptor head

    def train_mode(self, sample_dict: dict) -> torch.Tensor:
        if torch.cuda.is_available():
            sample_dict['image'] = sample_dict['image'].to('cuda')
            sample_dict['warped_image'] = sample_dict['warped_image'].to('cuda')
            sample_dict['label'] = sample_dict['label'].to('cuda')
            sample_dict['warped_label'] = sample_dict['warped_label'].to('cuda')
        batch_size, numHomoIter = sample_dict['warped_image'].shape[:2]
        (_, semi), (_, desc) = self.forward(sample_dict['image']).items()
        det_loss = detector_loss(sample_dict['label'], semi, mask=sample_dict['valid_mask'])
        det_loss_warp, desc_loss_warp = [], []
        for i in range(batch_size):  # iterate over batch_size
            # warped image is having dimension [Batch_size, NumOfHomoIter, height, width]
            (_, semi_warp), (_, desc_warp) = self.forward(sample_dict['warped_image'][i, :, :, :].unsqueeze(1)).items()
            det_loss_warp.append(detector_loss(sample_dict['warped_label'][i, :, :, :], output=semi_warp,
                                               mask=sample_dict['warped_mask'][i, :, :, :]))
            desc_loss_warp.append(
                descriptor_loss_2(desc[i, :, :, :], desc_warp, homography=sample_dict['homography'][i, :, :, :],
                                  margin_neg=0.2, margin_pos=1.0, lambda_d=250, threshold=8,
                                  valid_mask=sample_dict['warped_mask'][i, :, :, :]))
        det_loss_total = det_loss + sum(det_loss_warp) + sum(desc_loss_warp)
        return det_loss_total


def load_model(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    print('loading model: SuperPointNet ..............')
    if checkpoint_path[-4:] == '.pth':  # if only state_dict ie, weights of the file are stored
        model.load_state_dict(torch.load(checkpoint_path))
        epoch = epoch
        optimizer = optimizer
    else:  # if all data about training is available
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, epoch, optimizer
