import torch
from utils import detector_loss, descriptor_loss_2, flattenDetection, nms_fast
import numpy as np


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

    def train_mode(self, sample_dict: dict) -> dict:
        if torch.cuda.is_available():
            sample_dict['image'] = sample_dict['image'].to('cuda')
            sample_dict['warped_image'] = sample_dict['warped_image'].to('cuda')
            sample_dict['label'] = sample_dict['label'].to('cuda')
            sample_dict['warped_label'] = sample_dict['warped_label'].to('cuda')
        batch_size, numHomoIter = sample_dict['warped_image'].shape[:2]
        (_, semi), (_, desc) = self.forward(sample_dict['image']).items()
        (_, det_loss), (_, det_iou) = detector_loss(sample_dict['label'], semi, mask=sample_dict['valid_mask']).items()
        det_loss_warp, desc_loss_warp = [], []
        for i in range(batch_size):  # iterate over batch_size
            # warped image is having dimension [Batch_size, NumOfHomoIter, height, width]
            (_, semi_warp), (_, desc_warp) = self.forward(sample_dict['warped_image'][i, :, :, :].unsqueeze(1)).items()
            det_loss_warp_dict = detector_loss(sample_dict['warped_label'][i, :, :, :], output=semi_warp,
                                               mask=sample_dict['warped_mask'][i, :, :, :])
            det_loss_warp.append(det_loss_warp_dict['loss'])
            det_iou = torch.add(det_iou, det_loss_warp_dict['iou']) / 2
            desc_loss_warp.append(
                descriptor_loss_2(desc[i, :, :, :], desc_warp, homography=sample_dict['homography'][i, :, :, :],
                                  margin_neg=0.2, margin_pos=1.0, lambda_d=250, threshold=8,
                                  valid_mask=sample_dict['warped_mask'][i, :, :, :]))
        det_loss_total = det_loss + sum(det_loss_warp) + sum(desc_loss_warp)
        return {'loss': det_loss_total, 'iou': det_iou}

    def eval_mode(self, image: np.ndarray, conf_threshold: float, H: int, W: int, dist_thresh: float) -> tuple:
        with torch.no_grad():
            (_, semi), (_, desc) = self.forward(
                torch.from_numpy(image[np.newaxis, np.newaxis, :, :]).to('cuda')).items()
            heatmap = flattenDetection(semi).cpu().numpy().squeeze()
            xs, ys = np.where(heatmap >= conf_threshold)
            if len(xs) == 0:
                return np.zeros((3, 0)), None, None
            pts = np.zeros((3, len(xs)))
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[xs, ys]
            pts, _ = nms_fast(pts, H, W, dist_thresh=dist_thresh)
            inds = np.argsort(-pts[2, :])  # sort by confidence
            bord = 4  # border to remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]
            #  -- process descriptor
            D = desc.shape[1]
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                samp_pts = torch.from_numpy(pts[:2, :].copy())
                samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
                samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
                samp_pts = samp_pts.transpose(0, 1).contiguous()
                samp_pts = samp_pts.view(1, 1, -1, 2)
                samp_pts = samp_pts.float()
                if self.cuda:
                    samp_pts = samp_pts.cuda()
                desc = torch.nn.functional.grid_sample(desc, samp_pts)
                desc = desc.data.cpu().numpy().reshape(D, -1)
                desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap


def detector_post_processing(semi: torch.Tensor, conf_threshold=0.015, NMS_dist=1, ret_heatmap=False,
                             limit_detection=600) -> np.ndarray:
    """
    :param semi - Output prediction from SuperpointNet with shape (65, Hc, Wc) - takes only one image at once
    :param conf_threshold - Detector confidence threshold to be applied
    :param NMS_dist - Correct distance used for Non maximal suppression
    :param ret_heatmap - returns heatmap with size (Hc x 8, Wc x 8)
    :param limit_detection - total no. of detection which needs to be considered utmost.
    """
    assert len(semi.squeeze().shape) == 3
    with torch.no_grad():
        SoftMax = torch.nn.Softmax(dim=0)  # apply softmax on the channel dimension with 65
        soft_output = SoftMax(semi.squeeze())
        soft_output = soft_output[:-1, :, :]
        pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=8)
        heatmap = pixel_shuffle(soft_output).to('cpu').numpy().squeeze()
        if ret_heatmap:
            return heatmap
        xs, ys = np.where(heatmap >= conf_threshold)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        H, W = heatmap.shape
        pts, _ = nms_fast(pts, heatmap.shape[0], heatmap.shape[1], dist_thresh=NMS_dist)
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = 4  # we consider 4 pixels from all the boundaries as rejected
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        if pts.shape[1] > limit_detection:
            pts = pts[:, :limit_detection]
    return pts


def load_model(checkpoint_path: str, model: torch.nn.Module):
    print('loading model: SuperPointNet ..............')
    if checkpoint_path[-4:] == '.pth':  # if only state_dict ie, weights of the file are stored
        model.load_state_dict(torch.load(checkpoint_path))
    elif checkpoint_path[-4:] == '.tar':
        weight_dict = torch.load(checkpoint_path)
        model.load_state_dict(weight_dict['model_state_dict'])
    else:  # if all data about training is available
        model = torch.load(checkpoint_path)
        # model.load(checkpoint)
    return model


class SuperPointNetBatchNorm(SuperPointNet):
    """ Pytorch definition of SuperPoint Network with added Batch Normalization. From Daniel De Tone Implementation"""

    def __init__(self):
        super(SuperPointNetBatchNorm, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn1 = torch.nn.BatchNorm2d(num_features=c1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=c2)
        self.bn3 = torch.nn.BatchNorm2d(num_features=c3)
        self.bn4 = torch.nn.BatchNorm2d(num_features=c4)

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
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.bn3(x)
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.bn4(x)
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return {'semi': semi, 'desc': desc}  # semi is the detector head and desc is the descriptor head


class double_conv(torch.nn.Module):
    """"(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class SuperPointNet_gauss2(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, subpixel_channel=1):
        super(SuperPointNet_gauss2, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        self.relu = torch.nn.ReLU(inplace=True)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bnPa = torch.nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bnPb = torch.nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bnDa = torch.nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bnDb = torch.nn.BatchNorm2d(d1)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return {'semi': semi, 'desc': desc}

    def eval_mode(self, image: np.ndarray, conf_threshold: float, H: int, W: int, dist_thresh: float, top_k=300) -> \
            tuple:
        with torch.no_grad():
            (_, semi), (_, desc) = self.forward(
                torch.from_numpy(image[np.newaxis, np.newaxis, :, :]).to('cuda')).items()
            heatmap = flattenDetection(semi).cpu().numpy().squeeze()
            xs, ys = np.where(heatmap >= conf_threshold)
            if len(xs) == 0:
                return np.zeros((3, 0)), None, None
            pts = np.zeros((3, len(xs)))
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[xs, ys]
            pts, _ = nms_fast(pts, H, W, dist_thresh=dist_thresh)
            inds = np.argsort(-pts[2, :])  # sort by confidence
            bord = 4  # border to remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]
            if pts.shape[1] > top_k:
                pts = pts[:, :top_k]
            #  -- process descriptor
            D = desc.shape[1]
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                samp_pts = torch.from_numpy(pts[:2, :].copy())
                samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
                samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
                samp_pts = samp_pts.transpose(0, 1).contiguous()
                samp_pts = samp_pts.view(1, 1, -1, 2)
                samp_pts = samp_pts.float()
                if self.cuda:
                    samp_pts = samp_pts.cuda()
                desc = torch.nn.functional.grid_sample(desc, samp_pts)
                desc = desc.data.cpu().numpy().reshape(D, -1)
                desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap
