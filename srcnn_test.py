import torch,torchvision,os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as Data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from PIL import Image
from skimage.measure import compare_psnr
#from webdnn.frontend.pytorch import PyTorchConverter
#from webdnn.backend import generate_descriptor

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 5, 1, 2, bias=True)
        )
    def forward(self, x):
        return self.main(x)

if __name__ == "__main__":
    model = SRCNN()
    model.load_state_dict(torch.load("net.pth"))
#    dummy_input = torch.autograd.Variable(torch.randn(1, 3, 96, 96)).cuda()
#    graph = PyTorchConverter().convert(model.cuda(), dummy_input)
#    exec_info = generate_descriptor("webgl", graph)
#    exec_info.save("./output_model")
    pic = Image.open("manga.png").convert("RGB")
    pic1 = pic.resize((pic.size[0] // 2, pic.size[1] // 2), Image.ANTIALIAS)
    pic1 = pic1.resize((pic.size[0], pic.size[1]), Image.ANTIALIAS)
    groundTruth = transforms.ToTensor()(pic)
    input_ = transforms.ToTensor()(pic1)
    a,b,c = list(input_.size())
    input_ = input_.view(1, a, b, c)
    groundTruth = groundTruth.view(1, a, b, c)
    output = model(Variable(input_))
    print("PSNR: ",compare_psnr(groundTruth.numpy(), output.data.cpu().numpy()))
    vutils.save_image(output.data, nrow=1, filename="manga_hr.png")

