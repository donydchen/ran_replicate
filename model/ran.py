import torch 
from .base_model import BaseModel 
from . import model_utils


class RANModel(BaseModel):
    """docstring for RANModel"""
    def __init__(self):
        super(RANModel, self).__init__()
        self.name = "RAN"

    def initialize(self, opt):
        super(RANModel, self).initialize(opt)

        self.net_recog = model_utils.define_recog(self.opt.img_nc, self.opt.aus_nc, self.opt.which_model_netR, \
                            init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids, \
                            backend_pretrain=self.opt.backend_pretrain)
        self.models_name.append('recog')

        if self.is_train:
            self.n_dis = self.opt.exp_nc

            net_dis_list = model_utils.define_multi_dis(self.opt.aus_nc, self.n_dis, self.opt.hidden_nc_list, \
                            init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids)
            for idx in range(self.n_dis):
                setattr(self, "net_dis%d" % idx, net_dis_list[idx])
                self.models_name.append("dis%d" % idx)

        if self.opt.load_epoch > 0:
            self.load_ckpt(self.opt.load_epoch)

    def setup(self):
        super(RANModel, self).setup()
        if self.is_train:
            # setup optimizer
            self.optim_recog = torch.optim.Adam(self.net_recog.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optims.append(self.optim_recog)
            for idx in range(self.n_dis):
                setattr(self, "optim_dis%d" % idx, torch.optim.Adam(getattr(self, "net_dis%d" % idx).parameters(), \
                        lr=self.opt.lr, betas=(self.opt.beta1, 0.999)))
                self.optims.append(getattr(self, "optim_dis%d" % idx))

            # setup schedulers
            self.schedulers = [model_utils.get_scheduler(optim, self.opt) for optim in self.optims]

    def feed_batch(self, batch):
        # {'pseudo_aus': pseudo_aus, 'pseudo_exp': pseudo_exp, 'img_exp': img_exp, 'img': img_tensor}
        self.img = batch['img'].to(self.device)
        if self.is_train:
            self.img_exp = batch['img_exp'].to(self.device)
            self.pseudo_exp = batch['pseudo_exp'].to(self.device)
            self.pseudo_aus = batch['pseudo_aus'].type(torch.FloatTensor).to(self.device)

    def forward(self):
        self.gen_aus = self.net_recog(self.img)

    def backward_dis(self):
        # use label to select discriminator...I can't find an efficient way
        pred_real = []
        for idx in range(self.pseudo_exp.size[0]):
            cur_dis = getattr(self, "net_dis%d" % self.pseudo_exp.detach().cpu().int().numpy()[idx])
            pred_real.append(cur_dis(self.pseudo_aus[idx]))
        self.loss_dis_real = self.criterionGAN(pred_real, True)

        pred_fake = []
        for idx in range(self.img_exp.size[0]):
            cur_dis = getattr(self, "net_dis%d" % self.img_exp.detach().cpu().int().numpy()[idx])
            pred_fake.append(cur_dis(self.gen_aus[idx].detach()))  # stop backward to recognition net
        self.loss_dis_fake = self.criterionGAN(pred_fake, False)

        self.loss_dis = self.loss_dis_real + self.loss_dis_fake
        self.loss_dis.backward()

    def backward_recog(self):
        pred_fake = []
        for idx in range(self.img_exp.size[0]):
            cur_dis = getattr(self, "net_dis%d" % self.img_exp.detach().cpu().int().numpy()[idx])
            pred_fake.append(cur_dis(self.gen_aus[idx]))
        self.loss_recog = self.criterionGAN(pred_fake, True)

        self.loss_recog.backward()

    def optimize_paras(self, train_recog=True):
        self.forward()
        # update discriminator
        for idx in range(self.n_dis):
            self.set_requires_grad(getattr(self, "net_dis%d" % idx), True)
            getattr(self, "net_dis%d" % idx).zero_grad()
        self.backward_dis()
        for idx in range(self.n_dis):
            getattr(self, "net_dis%d" % idx).step()

        # update recognition net if needed
        if train_recog:
            for idx in range(self.n_dis):
                self.set_requires_grad(getattr(self, "net_dis%d" % idx), False)
            self.net_recog.zero_grad()
            self.backward_recog()
            self.net_recog.step()
                  
    def save_ckpt(self, epoch):
        save_models_name = ['recog']
        save_models_name.extend(["dis%d" % idx for idx in range(self.n_dis)])
        return super(RANModel, self).save_ckpt(epoch, save_models_name)

    def load_ckpt(self, epoch):
        # load the specific part of networks
        load_models_name = ['recog']
        if self.is_train:
            load_models_name.extend(["dis%d" % idx for idx in range(self.n_dis)])
        return super(RANModel, self).load_ckpt(epoch, load_models_name)

    def clean_ckpt(self, epoch):
        models_name = ['recog']
        models_name.extend(["dis%d" % idx for idx in range(self.n_dis)])
        return super(RANModel, self).clean_ckpt(epoch, models_name)

    def get_latest_losses(self):
        get_losses_name = ['dis_fake', 'dis_real', 'recog']
        return super(RANModel, self).get_latest_losses(get_losses_name)

    def get_latest_visuals(self):
        visuals_name = ['img']
        return super(RANModel, self).get_latest_visuals(visuals_name)
        