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

#configures
dataroot="/Users/DongSky/Downloads/faces/"
outf="."
imgSize=96
learning_rate=1e-4
batch_size=128
channel=3
workers=1
epoches=100
beta1=0.9
model_out = "./models"
if not os.path.exists(model_out):
    os.makedirs(model_out)
#####
def default_loader(path):
    return Image.open(path).convert('RGB')
class FacesDataset(torch.utils.data.Dataset):
    def __init__(self,path='faces/',transform=transforms.ToTensor(), target_transform=None, loader=default_loader):
        lst_hr = os.listdir(path+"hr/")
        lst_lr = os.listdir(path+"lr/")
        imgs = []
        for i in range(len(lst_hr)):
                imgs.append((path+"hr/"+lst_hr[i],path+"lr/"+lst_lr[i]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        HR, LR = self.imgs[index]
        img = self.loader(LR)
        label = self.loader(HR)
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
        return img,label

    def __len__(self):
        return len(self.imgs)
        pass
dataset=FacesDataset(dataroot)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=workers)
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # self.main = nn.Sequential(
        #     nn.Conv2d(channel, 64, 9, 1, 4, bias=True),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 32, 3, 1, 1, bias=True),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 3, 5, 1, 2, bias=True),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(True)
        # )
        self.main = nn.Sequential(
            nn.Conv2d(channel, 64, 9, 1, 4, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 5, 1, 2, bias=True)
        )
    def forward(self, x):
        return self.main(x)

def init(l):
    if isinstance(l, nn.ConvTranspose2d) or isinstance(l, nn.Conv2d):
        l.weight.data.normal_(0.0,0.02)
        l.bias.data.zero_()
    elif isinstance(l, nn.BatchNorm2d):
        l.weight.data.normal_(1.0,0.02)
        l.bias.data.zero_()
cnt = 0
srcnn = SRCNN()
srcnn.apply(init)
if torch.cuda.is_available():
  srcnn.cuda()
criter = nn.MSELoss()
x = torch.FloatTensor(batch_size, 3, imgSize, imgSize)
label = torch.FloatTensor(batch_size, 3, imgSize, imgSize)
if torch.cuda.is_available():
    x, label = x.cuda(), label.cuda()
opt = optim.Adam(srcnn.parameters(),betas=(0.9, 0.999), lr=learning_rate)
sche = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
for epoch in range(epoches):
    for i,data in enumerate(dataloader, 0):
        cnt += 1
        opt.zero_grad()
        lr, hr = data
        if torch.cuda.is_available():
            lr, hr = lr.cuda(), hr.cuda()
        x.resize_as_(lr).copy_(lr)
        label.resize_as_(hr).copy_(hr)
        xv = Variable(x)
        labelv = Variable(label)
        output = srcnn(xv)
        eriri = criter(output, labelv)
        eriri.backward()
        opt.step()
        if i % 100 == 0:
          print('[%d/%d][%d/%d] Loss: %.6f'
              % (epoch, epoches, i, len(dataloader),
                 eriri.data[0]))
        if cnt % 1000 == 0:
            vutils.save_image(labelv.data,
                              '%s/real_samples_epoch_%03d_iter_%04d.png' % (outf, epoch, i),
                              normalize=True)
            vutils.save_image(output.data,
                              '%s/fake_samples_epoch_%03d_iter_%04d.png' % (outf, epoch, i),
                              normalize=True)
            vutils.save_image(lr,
                              '%s/init_samples_epoch_%03d_iter_%04d.png' % (outf, epoch, i),
                              normalize=True)
            # upload('%s/real_samples_epoch_%03d_iter_%04d.png' % (outf, epoch, i))
            # upload('%s/fake_samples_epoch_%03d_iter_%04d.png' % (outf, epoch, i))
            # upload('%s/init_samples_epoch_%03d_iter_%04d.png' % (outf, epoch, i))
    sche.step()
    torch.save(srcnn.state_dict(), '%s/net_epoch_%d.pth' % (model_out, epoch))
    # upload('%s/net_epoch_%d.pth' % (model_out, epoch))
