import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse

import utils

import torch
import torchvision.transforms.functional as TF
import cyclegan_networks as cycnet

class Evaluator():
    def __init__(self, ckptdir, net_G_arg='unet_512', in_size=512):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ckptdir = ckptdir
        self.net_G_arg = net_G_arg
        self.in_size = in_size
        self.net_G = self.load_model()

    def run_eval(self, img_mix, img_dir, frame):
        print('running evaluation...')
        running_psnr = []
        running_ssim = []

        #img_mix = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        #img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
        #what to do with gt??

        # we recommend to use TF.resize since it was also used during trainig
        # You may also try cv2.resize, but it will produce slightly different results
        img_mix = TF.resize(TF.to_pil_image(img_mix), [self.in_size, self.in_size])
        img_mix = TF.to_tensor(img_mix).unsqueeze(0)

        #gt = TF.resize(TF.to_pil_image(gt), [args.in_size, args.in_size])
        #gt = TF.to_tensor(gt).unsqueeze(0)

        with torch.no_grad():
            G_pred = self.net_G(img_mix.to(self.device))[:, 0:3, :, :]
            G_pred2 = self.net_G(img_mix.to(self.device))[:, 3:6, :, :]

        G_pred = np.array(G_pred.cpu().detach())
        G_pred = G_pred[0, :].transpose([1, 2, 0])
        
        G_pred2 = np.array(G_pred2.cpu().detach())
        G_pred2 = G_pred2[0, :].transpose([1, 2, 0])

        # gt = np.array(gt.cpu().detach())
        # gt = gt[0, :].transpose([1, 2, 0])

        img_mix = np.array(img_mix.cpu().detach())
        img_mix = img_mix[0, :].transpose([1, 2, 0])
        
        G_pred[G_pred > 1.0] = 1.0
        G_pred[G_pred < 0] = 0
        
        G_pred2[G_pred2 > 1.0] = 1.0
        G_pred2[G_pred2 < 0] = 0

        psnr = 0#utils.cpt_rgb_psnr(G_pred, gt, PIXEL_MAX=1.0)
        ssim = 0#utils.cpt_rgb_ssim(G_pred, gt)
        running_psnr.append(psnr)
        running_ssim.append(ssim)

        outfile_fiducials = f"{img_dir}/fiducials/{str(frame)}.png"
        outfile_removed = f"{img_dir}/removed/{str(frame)}.png"
        plt.imsave(outfile_fiducials, G_pred)
        plt.imsave(outfile_removed, G_pred2)

        return G_pred, G_pred2

    def load_model(self):
        net_G = cycnet.define_G(
                input_nc=3, output_nc=6, ngf=64, netG=self.net_G_arg, use_dropout=False, norm='none').to(self.device)
        print('loading the best checkpoint...')
        checkpoint = torch.load(os.path.join(self.ckptdir, 'best_ckpt.pt'), map_location=self.device)
        net_G.load_state_dict(checkpoint['model_G_state_dict'])
        net_G.to(self.device)
        net_G.eval()

        return net_G

