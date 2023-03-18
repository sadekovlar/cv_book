import os
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_video, write_video
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import raft_large, raft_small

file_name = 'nhd.002.001.left.avi'
video_path = os.path.join('..','data', 'optical_flow', file_name)

#weights = Raft_Large_Weights.DEFAULT
weights = Raft_Small_Weights.DEFAULT
transforms = weights.transforms()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

#model = raft_large(weights=weights, progress=False).to(device)
model = raft_small(weights=weights, progress=False).to(device)
model = model.eval()

def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)

frames, _, _ = read_video(str(video_path), output_format="TCHW")
result = []
size = frames.shape[0]
for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
    img1, img2 = preprocess(img1, img2)
    img1 = img1[None, :, :, :]
    img2 = img2[None, :, :, :]
    with torch.no_grad():
        list_of_flows = model(img1.to(device), img2.to(device))
    predicted_flow = list_of_flows[-1][0]
    flow_img = flow_to_image(predicted_flow).to("cpu")
    print(f"Done {i+1}/{size-1}")
    result.append(flow_img.permute(1, 2, 0))

res_tensor = torch.stack(result)
write_video('RAFT video/RAFT.avi', res_tensor, 2)
print("done")
