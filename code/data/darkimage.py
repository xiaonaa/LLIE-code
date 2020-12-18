import os
from data import srdata

class DarkImage(srdata.SRData):
    def __init__(self, args, name='DarkImage', train=True, benchmark=False):
        super(DarkImage, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DarkImage, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DarkImage, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DarkImage_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DarkImage_train_LR')

