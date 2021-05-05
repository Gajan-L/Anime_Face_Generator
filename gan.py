from tqdm import tqdm
import torch
import torchvision as tv
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from torchnet.meter import AverageValueMeter


# define parameters in class Config
class Config(object):
    data_path = './input2'
    virs = "result2"
    num_workers = 4  # numbers of thread
    img_size = 96  # size of input images
    batch_size = 256  # size of each batch
    max_epoch = 1000   # maximum epochs for training
    lr1 = 2e-4  # learning rate of generator network
    lr2 = 2e-4  # learning rate of discriminator network
    beta1 = 0.5  # beta value of Adam optimizer
    gpu = True  # using GPU for training if ture
    nz = 100  # dim of noise
    ngf = 64  # convolution kernels of generator
    ndf = 64  # convolution kernels of discriminator

    # model storage
    save_path = './imgs3'  # storing path of model
    # discriminator is trained for more times than generator
    d_every = 1  # train discriminator for every batch
    g_every = 5  # train generator for every 5 batches
    save_every = 10  # save the training models for every 10 epochs
    netd_path = None
    netg_path = None

    # generated image
    gen_img = "result.png"
    gen_num = 64  # generate 64 images
    gen_search_num = 512
    gen_mean = 0    # mean of generated noise
    gen_std = 1     # variance of generated noise


# construct the class Config
opt = Config()


# generator network
class NetG(nn.Module):
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.ngf = opt.ngf
        self.Gene = nn.Sequential(
            # Assumed that the input date is vector with the shape of opt.nz * 1 * 1
            # output = (input - 1) * stride + output_padding - 2 * padding + kernel_size
            nn.ConvTranspose2d(in_channels=opt.nz, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),

            # input: 4 * 4 * ngf * 8
            nn.ConvTranspose2d(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2, padding=1, bias =False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),

            # input: 8 * 8 * ngf * 4
            nn.ConvTranspose2d(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),

            # input: 16 * 16 * ngf * 2
            nn.ConvTranspose2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            # input: 32 * 32 * ngf
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),

            # Tanh converge faster than sigmoid,much more slower than relu
            # range of output: [-1,1]
            # mean of output: 0
            nn.Tanh(),

        )
        # output: 96 * 96 * 3

    def forward(self, x):
        return self.Gene(x)

# discriminator network
class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()

        self.ndf = opt.ndf
        self.Discrim = nn.Sequential(
            # convolutional layer
            # input:(bitch_size, 3, 96, 96), bitch_size = numbers of training samples for each train
            # output:(bitch_size, ndf, 32, 32), (96 - 5 +2 *1)/3 + 1 =32
            nn.Conv2d(in_channels=3, out_channels=self.ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),

            # input: (ndf, 32, 32)
            nn.Conv2d(in_channels= self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),

            # input:(ndf * 2, 16, 16)
            nn.Conv2d(in_channels= self.ndf * 2, out_channels=self.ndf * 4, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),

            # input:(ndf * 4, 8, 8)
            nn.Conv2d(in_channels= self.ndf *4, out_channels=self.ndf *8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf *8),
            nn.LeakyReLU(0.2, True),

            # input:(ndf * 8, 4, 4)
            # output:(1, 1, 1)
            nn.Conv2d(in_channels=self.ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.Discrim(x).view(-1)


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # choosing device for training
    if opt.gpu:
        device = torch.device("cuda")
        print('using GPU')
    else:
        device = torch.device('cpu')
        print('using CPU')

    # data preprocessing
    transforms = tv.transforms.Compose([
        # 3*96*96
        tv.transforms.Resize(opt.img_size),   # resize images to img_size* img_size
        tv.transforms.CenterCrop(opt.img_size),

        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(root=opt.data_path, transform=transforms)

    dataloader = DataLoader(
        dataset,  # loading dataset
        batch_size=opt.batch_size,  # setting batch size
        shuffle=True,  # choosing if shuffle or not
        num_workers=opt.num_workers,  # using multiple threads for processing
        drop_last=True  # if true, drop the last batch if the batch is not fitted the size of batch size
    )

    # initialize network
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc: storage

    # torch.load for loading models
    if opt.netg_path:
        netg.load_state_dict(torch.load(f=opt.netg_path, map_location=map_location))
    if opt.netd_path:
        netd.load_state_dict(torch.load(f=opt.netd_path, map_location=map_location))

    # move models to device
    netd.to(device)
    netg.to(device)

    # Adam optimizer
    optimize_g = torch.optim.Adam(netg.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
    optimize_d = torch.optim.Adam(netd.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))

    # BCEloss:-w(ylog x +(1 - y)log(1 - x))
    # y: real label，x: score from discriminator using sigmiod( 1: real, 0: fake)
    criterions = nn.BCELoss().to(device)

    # define labels
    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)

    # generate a noise with the distribution of N(1,1)，dim = opt.nz，size = opt.batch_size
    noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    # for generating images when saving models
    fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()
    write = SummaryWriter(log_dir=opt.virs, comment='loss')

    # training
    for epoch in range(opt.max_epoch):
        for ii_, (img, _) in tqdm((enumerate(dataloader))):
            real_img = img.to(device)

            # begin training
            # train discriminator for every d_every batches
            if ii_ % opt.d_every == 0:
                # clear optimizer gradient
                optimize_d.zero_grad()

                output = netd(real_img)
                error_d_real = criterions(output, true_labels)
                error_d_real.backward()

                # generate fake image
                noises = noises.detach()
                # generate fake images data using noises
                fake_image = netg(noises).detach()
                # discriminator discriminate fake images
                output = netd(fake_image)
                error_d_fake = criterions(output, fake_labels)
                error_d_fake.backward()

                optimize_d.step()

                error_d = error_d_fake + error_d_real
                errord_meter.add(error_d.item())

            # train generator for every g_every batches
            if ii_ % opt.g_every == 0:
                optimize_g.zero_grad()
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_image = netg(noises)
                output = netd(fake_image)
                error_g = criterions(output, true_labels)
                error_g.backward()

                optimize_g.step()

                errorg_meter.add(error_g.item())

        # draw graph of loss
        if ii_ % 5 == 0:
            write.add_scalar("Discriminator_loss", errord_meter.value()[0])
            write.add_scalar("Generator_loss", errorg_meter.value()[0])

        # saving models for save_every batches
        if (epoch + 1) % opt.save_every == 0:
            fix_fake_image = netg(fix_noises)
            tv.utils.save_image(fix_fake_image.data[:64], "%s/%s.png" % (opt.save_path, epoch), normalize=True)

            torch.save(netd.state_dict(),  'imgs3/' + 'netd_{0}.pth'.format(epoch))
            torch.save(netg.state_dict(),  'imgs3/' + 'netg_{0}.pth'.format(epoch))
            errord_meter.reset()
            errorg_meter.reset()

    write.close()


@torch.no_grad()
def generate(**kwargs):
    # generate images using trained models

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device("cuda") if opt.gpu else torch.device("cpu")

    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    map_location = lambda storage, loc: storage

    netd.load_state_dict(torch.load('imgs3/netd_999.pth', map_location=map_location), False)
    netg.load_state_dict(torch.load('imgs3/netg_999.pth', map_location=map_location), False)
    netd.to(device)
    netg.to(device)

    # generate images
    noise = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std).to(device)
    noise.to(device)
    fake_image = netg(noise)
    score = netd(fake_image).detach()

    # select the images with higher score
    indexs = score.topk(opt.gen_num)[1]

    result = []

    for ii in indexs:
        result.append(fake_image.data[ii])

    # saving generated images
    tv.utils.save_image(torch.stack(result), opt.gen_img, normalize=True, range=(-1, 1))

def main():
    # training models
    train()
    # generating images
    generate()

if __name__ == '__main__':
    main()
