from utils import sample_homography, inv_warp_image_batch, warp_points
import torch
import numpy as np
import yaml

# config_file = "joint_training.yaml"
# with open(config_file) as file:
#     config = yaml.full_load(file)
#
# warped_pair_params = config['data']['augmentation']['homographic']['homographies']['params']
# coords = torch.stack(torch.meshgrid(torch.linspace(0, 3, 3), torch.linspace(0, 3, 3)), dim=2).type(torch.float32)
# coords = coords.transpose(1, 0)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# homography = torch.from_numpy(sample_homography(np.array([2, 2]), shift=-1, **warped_pair_params)).type(torch.float32)
# warped_coord = warp_points(coords.reshape([-1, 2]), homography)
# coords = coords.reshape([-1, 2])
# norm = torch.linalg.norm(warped_coord - coords, dim=-1)
# mask = norm <= 16
# desc_1 = torch.rand(size=(9, 256), dtype=torch.float32).unsqueeze(1)
# desc_2 = torch.rand(size=(9, 256), dtype=torch.float32).unsqueeze(2)
# desc_prod = desc_1 * desc_2
# desc_prod_2 = torch.matmul(desc_1, desc_2)
#
# print(warped_coord)
for i in range(10):
    if i == 5:
        print(i, '5 was here')
        continue
    print(i, 'It is not 5')
