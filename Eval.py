#%%
import os
import re
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src import utils
from src.transformer_net import TransformerNet

#%% Global params
MODEL_NAME = "rain_512_epoch_1_cw_1.0_sw_1000000.0"
STYLE = "rain_512"
TEST_IMG = "hsuan"
TEST_SCALE = 1
OUTPUT_IMG = TEST_IMG + "_" + STYLE

MODEL_DIR = os.path.join("model", STYLE, MODEL_NAME) + ".pth"
TEST_IMG_DIR = os.path.join("images/1-input", TEST_IMG) + ".jpg"
OUT_IMG_DIR = os.path.join("images/3-output", OUTPUT_IMG) + ".jpg"

#%%
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    content_image = utils.load_image(TEST_IMG_DIR, scale=TEST_SCALE)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        transformerNet = TransformerNet()
        state_dict = torch.load(MODEL_DIR)['state_dict']
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        transformerNet.load_state_dict(state_dict)
        transformerNet.to(device)
        
        output = transformerNet(content_image).cpu()
        
        utils.save_image(OUT_IMG_DIR, output[0])
        
        if str(device) == 'cuda':
            torch.cuda.empty_cache()
        
def loss_plot():
    content_loss = torch.load(MODEL_DIR)['content_loss']
    style_loss = torch.load(MODEL_DIR)['style_loss']
    content_weight = torch.load(MODEL_DIR)['content_weight']
    style_weight = torch.load(MODEL_DIR)['style_weight']
    
    plt.figure(0)
    plt.plot(content_loss, label="content loss")
    plt.plot(style_loss, label="style loss")
    plt.legend()
    title_str = f"Loss, cw = {content_weight}, sw = {style_weight}"
    # title_str = f"Learning rate = 1e-4"
    plt.title(title_str)
    plt.xlabel("iterations")
    plt.show()
    
#%%
if __name__ == "__main__":
    evaluate()
    loss_plot()
# %%
