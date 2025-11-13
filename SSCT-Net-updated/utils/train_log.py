import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class TrainLogger:
    def __init__(self, savepath):
        self.savepath = savepath
        self.epochs = []
        self.losses = []
        self.psnrs = []
        self.sams = []

    def append(self, epoch, loss, psnr, sam):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.psnrs.append(psnr)
        self.sams.append(sam)
        self._plot()

    def _plot(self):
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(self.epochs, self.losses, label='loss')
        plt.xlabel('epoch'); plt.legend()
        plt.subplot(1,2,2)
        plt.plot(self.epochs, self.psnrs, label='psnr')
        plt.plot(self.epochs, self.sams, label='sam')
        plt.xlabel('epoch'); plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(self.savepath), exist_ok=True)
        plt.savefig(self.savepath)
        plt.close()
