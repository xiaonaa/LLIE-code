import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm


class Trainer():
    def __init__(self, args, loader, my_model_f, my_model_u, my_model_e,my_model_n, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model_f = my_model_f
        self.model_u = my_model_u
        self.model_e = my_model_e
        self.model_n = my_model_n
        self.loss = my_loss
        self.optimizer_f = utility.make_optimizer(args, self.model_f)
        self.optimizer_u = utility.make_optimizer(args, self.model_u)
        self.optimizer_e = utility.make_optimizer(args, self.model_e)
        self.optimizer_n = utility.make_optimizer(args, self.model_n)
        self.scheduler_f = utility.make_scheduler(args, self.optimizer_f)
        self.scheduler_u = utility.make_scheduler(args, self.optimizer_u)
        self.scheduler_e = utility.make_scheduler(args, self.optimizer_e)
        self.scheduler_n = utility.make_scheduler(args, self.optimizer_n)

        if self.args.load != '.':
            self.optimizer_f.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer_f.pt'))
            )
            self.optimizer_u.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer_u.pt'))
            )
            self.optimizer_e.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer_e.pt'))
            )
            self.optimizer_n.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer_n.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler_f.step()
            for _ in range(len(ckp.log)): self.scheduler_u.step()
            for _ in range(len(ckp.log)): self.scheduler_e.step()
            for _ in range(len(ckp.log)): self.scheduler_n.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler_f.step()
        self.scheduler_u.step()
        self.scheduler_e.step()
        self.scheduler_n.step()
        self.loss.step()
        epoch = self.scheduler_f.last_epoch + 1
        lr_i = self.scheduler_f.get_lr()[0]
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr_i))
        )
        self.loss.start_log()
        self.model_f.train()
        self.model_u.train()
        self.model_e.train()
        self.model_n.train()
        timer_data, timer_model_f = utility.timer(), utility.timer()
        for batch, (lr,hr, idx_scale) in enumerate(self.loader_train):
            lr,hr = self.prepare( lr,hr)
            timer_data.hold()
            timer_model_f.tic()
            self.optimizer_f.zero_grad()
            self.optimizer_u.zero_grad()
            self.optimizer_e.zero_grad()
            self.optimizer_n.zero_grad()
            map = self.model_f(lr, idx_scale)
            lrI = torch.cat([lr, map], 1)
            sr_u = self.model_u(lrI, idx_scale)
            sr_e = self.model_e(lrI, idx_scale)
            lr_n = torch.cat([lr, sr_u], 1)
            sr_n = self.model_n(lr_n, idx_scale)
            sr = sr_u * sr_e + sr_n
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer_u.step()
                self.optimizer_e.step()
                self.optimizer_f.step()
                self.optimizer_n.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model_f.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model_f.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler_f.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model_f.eval()
        self.model_u.eval()
        self.model_e.eval()
        self.model_n.eval()
        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr,hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr,hr = self.prepare(lr,hr)
                    else:
                        lr, = self.prepare(lr)

                    map = self.model_f(lr, idx_scale)
                    lrI = torch.cat([lr, map], 1)
                    sr_u = self.model_u(lrI, idx_scale)
                    sr_e = self.model_e(lrI, idx_scale)
                    lr_n = torch.cat([lr, sr_u], 1)
                    sr_n = self.model_n(lr_n, idx_scale)
                    sr = sr_u * sr_e + sr_n
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        # save_list.extend([lr,hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)

                if not self.args.test_only:
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            self.args.data_test,
                            scale,
                            self.ckp.log[-1, idx_scale],
                            best[0][idx_scale],
                            best[1][idx_scale] + 1
                        )
                    )
                    self.ckp.write_log(
                        'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
                    )
                    self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler_f.last_epoch + 1
            return epoch >= self.args.epochs

