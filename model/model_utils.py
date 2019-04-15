import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torchvision.models as torch_models


'''
Helper functions for model
Borrow tons of code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
'''


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], backend_pretrain=False):
    if len(gpu_ids) > 0:
        # print("gpu_ids,", gpu_ids)
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if not backend_pretrain:
        init_weights(net, init_type, gain=init_gain)
    return net


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def custom_resnet(model_name='resnet18', pretrained=False, **kwargs):
    if pretrained and 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
        # patch n_classes params
        n_classes = kwargs['num_classes']
        kwargs['num_classes'] = 1000

        model = getattr(torch_models, model_name)(**kwargs)
        pretrained_state_dict = torch.utils.model_zoo.load_url(model_urls[model_name])
        # load only existing feature
        pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict)
        print("[Info] Successfully load ImageNet pretrained parameters for %s." % model_name)
        # update fc layer
        print("Predict Class Number:", n_classes)
        model.fc = nn.Linear(pretrained_state_dict['fc.weight'].size(1), n_classes)
        init.xavier_normal_(model.fc.weight.data, 0.02)
        init.constant_(model.fc.bias.data, 0.0)
    else:
        model = getattr(torch_models, model_name)(pretrained, **kwargs)

    return model


def define_recog(input_nc, n_classes, which_model_netR, image_size, init_type='normal', init_gain=0.02, gpu_ids=[], backend_pretrain=False):
    net_recog = None
    if which_model_netR == 'default':
        net_recog = RecognitionNet(input_nc, n_classes, image_size)
    elif 'resnet' in which_model_netR:
        # input size 224
        net_recog = custom_resnet(which_model_netR, pretrained=backend_pretrain, num_classes=n_classes)
    else:
        raise NotImplementedError('Recognition model [%s] is not implemented' % which_model_netR)
    return init_net(net_recog, init_type, init_gain, gpu_ids, backend_pretrain)


def define_multi_dis(input_nc, n_dis, hidden_nc_list=None, init_type='normal', init_gain=0.02, gpu_ids=[]):
    dis_list = []
    for idx in range(n_dis):
        cur_dis = ThreeLayerDisNet(input_nc, hidden_nc_list)
        cur_dis = init_net(cur_dis, init_type, init_gain, gpu_ids)
        dis_list.append(cur_dis)
    return dis_list


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_type='wgan-gp', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_type = gan_type
        if self.gan_type == 'wgan-gp':
            self.loss = lambda x, y: -torch.mean(x) if y else torch.mean(x)
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'gan':
            self.loss = nn.BCELoss()
        else:
            raise NotImplementedError('GAN loss type [%s] is not found' % gan_type)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            target_tensor = target_is_real
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class RecognitionNet(nn.Module):
    """ Use linear function for recognition model as illustrated in the original paper.
        image_size: int"""
    def __init__(self, input_nc, n_classes, image_size):
        super(RecognitionNet, self).__init__()
        n_input = input_nc * image_size * image_size

        model = [nn.Linear(n_input, n_classes, bias=True),
                 nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, img):
        img = img.view(img.size(0), -1)
        pred_aus = self.model(img)
        return pred_aus


class ThreeLayerDisNet(nn.Module):
    """A three-layer feedforward net for discriminator
        hidden_nc_list: with 2 items"""
    def __init__(self, input_nc, hidden_nc_list=None):
        super(ThreeLayerDisNet, self).__init__()
        if hidden_nc_list is None:
            hidden_nc_list = [240, 240]  # set default as the vanilla GAN
        
        model = [nn.Linear(input_nc, hidden_nc_list[0]),
                 nn.ReLU(),
                 nn.Linear(hidden_nc_list[0], hidden_nc_list[1]),
                 nn.ReLU(),
                 nn.Linear(hidden_nc_list[1], 1),
                 nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, aus):
        out = self.model(aus)
        return out
        