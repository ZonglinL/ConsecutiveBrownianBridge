from flolpips import Flolpips
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_metric = Flolpips().to(device)

batch = 8
I0 = torch.rand(8, 3, 256, 448).to(device)
I1 = torch.rand(8, 3, 256, 448).to(device)
frame_dis = torch.rand(8, 3, 256, 448).to(device)
frame_ref = torch.rand(8, 3, 256, 448).to(device)

flolpips = eval_metric.forward(I0, I1, frame_dis, frame_ref)
print(flolpips)