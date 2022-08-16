import torch
import torch.nn as nn

# define Residual block in the Extractor
class ResBlock1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock1, self).__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

        self.short_cut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out


# define Residual block in the generator
class ResBlock2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock2, self).__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))

        self.short_cut = nn.Sequential(nn.BatchNorm2d(in_dim, affine=True),
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out


# define convolutiona block in the discriminator
class ConvBlock(nn.Module):
    def __init__(self,in_dim,out_dim,s=2):
        super(ConvBlock,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim,out_dim,kernel_size=4,stride=s,padding=1),
            nn.BatchNorm2d(out_dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

    def forward(self,x):
        out = self.layer(x)
        return out


# define deconvolution block in the generator
class DeconvBlock(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(DeconvBlock,self).__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(out_dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self,x):
        out = self.deconv_block(x)
        return out


# define Mapping class
class MappingBlock(nn.Module):
    def __init__(self):
        super(MappingBlock,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512,512),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

    def forward(self,x):
        out = self.layer(x)
        return out


# define Extractor
class Extractor(nn.Module):
    def __init__(self,z_dim=512):
        super(Extractor,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1)
        self.res_blocks = nn.Sequential(ResBlock1(64, 128),
                                        ResBlock1(128, 128))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(128, affine=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        # (N, 3, 256, 256) -> (N, 64, 128, 128)
        out = self.conv1(x)
        # (N, 64, 128, 128) -> (N, 128, 64, 64) -> (N, 128, 32, 32)
        out = self.res_blocks(out)
        # (N, 128, 32, 32) -> (N, 128, 32, 32)
        out = self.conv2(out)
        return out

# define G-Mapping class
class Mapping(nn.Module):
    def __init__(self):
        super(Mapping,self).__init__()
        self.layer = nn.Sequential(MappingBlock(),
                                   MappingBlock(),
                                   MappingBlock(),
                                   MappingBlock(),
                                   MappingBlock(),
                                   MappingBlock(),
                                   MappingBlock(),
                                   MappingBlock())

    def forward(self,x):
        out = self.layer(x)
        return out


# define Generator
class Generator(nn.Module):
    def __init__(self, init_weights=True):
        super(Generator,self).__init__()
        self.mapping = Mapping()
        self.extractor = Extractor(z_dim=512)
        self.fc = nn.Sequential(
            nn.Linear(512,1024),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32, affine=True),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.res_block = nn.Sequential(ResBlock2(256,256),
                                       ResBlock2(256,256),
                                       ResBlock2(256,256),
                                       ResBlock2(256,256),
                                       ResBlock2(256,256),
                                       ResBlock2(256,256),
                                       ResBlock2(256,256),
                                       ResBlock2(256,256),
                                       ResBlock2(256,256))
        self.deconv = nn.Sequential(DeconvBlock(256,128),
                                    DeconvBlock(128,64),
                                    DeconvBlock(64,32))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,3,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
        if init_weights:
            self.initialize_weights()

    def forward(self,x,z):
        x = self.extractor(x)              # (batch_size,128,32,32)
        #print('x.size',x.size())
        z = self.mapping(z)                # (batch_size,512)
        #print('z.size',z.size())
        out = self.fc(z)          # (batch_size,1024)
        #print('self.fc(z).size',out.size())
        out = out.view(out.size(0),1,32,32)         # (batch_size,1,32,32)
        #print('out.view(-1,3,32,32).size',out.size())
        out = self.conv1(out)      # (batch_size,128,32,32)
        #print('conv1(out).size',out.size())
        cat = torch.cat([out,x],dim=1)    # (batch_size,256,32,32)
        #cat = torch.cat([out,x],dim=1)   # (batch_size,256,32,32)
        #print('cat.size',cat.size())
        out = self.res_block(cat)  # (batch_size,256,32,32)
        #print('res_block.size',out.size())
        out = self.deconv(out)     # (batch_size,32,256,256)
        #print('deconv(out).size',out.size())
        out = self.conv2(out)      # (batch_size,3,256,256)
        #print('conv2(out).size',out.size())
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# define discriminator
class Discriminator(nn.Module):
    def __init__(self,init_weights=True):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Sequential(ConvBlock(3,64,s=1),       #(batch_size,64,256,256)
                                   ConvBlock(64,64,s=2),      #(batch_size,64,128,128)
                                   ConvBlock(64,128,s=1),     #(batch_size,128,128,128)
                                   ConvBlock(128,128,s=2),    #(batch_size,128,64,64)
                                   ConvBlock(128,256,s=1),    #(batch_size,256,64,64)
                                   ConvBlock(256,512,s=2),    #(batch_size,512,32,32)
                                   ConvBlock(512,512,s=2))    #(batch_size,512,16,16)
        self.conv2 = nn.Conv2d(512,1,kernel_size=3,stride=1,padding=1)
        if init_weights:
            self.initialize_weights()

    def forward(self,x):
        out = self.conv1(x)
        #print('conv1.size',out.size())
        out = self.conv2(out)
        #print('conv2.size',out.size())
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
