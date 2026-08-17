import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.nn import MSELoss
from torchinfo import summary
from tqdm import tqdm

from utils.util import generate_mask


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction='mean').to(self.device)

        if self.params.parallel:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)

        # torchinfo/ptflops will call model.forward().
        # Our foundation model returns fused features in forward() (for finetune/inference),
        # and uses forward_pretrain() for MAE reconstruction.
        # Wrap it so complexity tools measure the pretrain forward path.
        class _PretrainWrapper(torch.nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core
            def forward(self, x):
                recon_t, _, _, _, _, _ = self.core.forward_pretrain(x, mask_t=None, mask_f=None)
                return recon_t

        wrapper = _PretrainWrapper(self.model)

        summary(wrapper, input_size=(1, 16, 30, 200))

        macs, params = get_model_complexity_info(wrapper, (16, 30, 200), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                           weight_decay=self.params.weight_decay)

        if self.params.lr_scheduler=='CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler=='ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif self.params.lr_scheduler=='StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5*self.data_length, gamma=0.5
            )
        elif self.params.lr_scheduler=='MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10*self.data_length, 20*self.data_length, 30*self.data_length], gamma=0.1
            )
        elif self.params.lr_scheduler=='CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=0.001, step_size_up=self.data_length*5,
                step_size_down=self.data_length*2, mode='exp_range', gamma=0.9, cycle_momentum=False
            )


    @staticmethod
    def masked_mse_2dfeat(pred, tgt, mask_f):
        """Masked MSE for band features.

        Args:
            pred, tgt: (B, C, Bands, 2)
            mask_f: (B, C, Bands) with {0,1}
        """
        mf = (mask_f > 0).unsqueeze(-1).to(tgt.dtype)  # (B,C,B,1)
        mf = mf.expand_as(tgt)                          # (B,C,B,2)
        diff = (pred - tgt) ** 2
        return (diff * mf).sum() / (mf.sum() + 1e-8)


    @staticmethod
    def _masked_mse_band2(pred: torch.Tensor, tgt: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        """Masked MSE for SF branch with 2-D band features.

        pred, tgt: (B, C, Bands, 2)
        mask_f: (B, C, Bands) with {0,1}
        """
        # Expand mask to match the last feature dimension (=2)
        mf = (mask_f > 0).unsqueeze(-1).to(tgt.dtype)  # (B,C,B,1)
        mf = mf.expand_as(tgt)                         # (B,C,B,2)
        diff = (pred - tgt) ** 2
        return (diff * mf).sum() / (mf.sum() + 1e-8)


    def train(self):
        best_loss = 10000
        for epoch in range(self.params.epochs):
            losses = []
            for x in tqdm(self.data_loader, mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)/100

                if not torch.isfinite(x).all():
                    print("Skip batch: input contains NaN or Inf")
                    continue
                
                if self.params.need_mask:
                    bz, ch_num, patch_num, patch_size = x.shape

                    # Random band mask for SF branch: shape (B, C, Bands)
                    band_num = len(str(getattr(self.params, 'sf_bands', '0.5-4,4-8,8-13,13-30,30-45')).split(','))
                    mask_f = generate_mask(
                        bz, ch_num, band_num,
                        mask_ratio=getattr(self.params, 'freq_mask_ratio', self.params.mask_ratio),
                        device=self.device,
                    )

                    # Random temporal-spatial mask only
                    mask = generate_mask(
                        bz, ch_num, patch_num,
                        mask_ratio=self.params.mask_ratio,
                        device=self.device,
                    )
                    recon_t, recon_f, feats_t, feats_f, _, freq_target = self.model.forward_pretrain(
                        x, mask_t=mask, mask_f=mask_f
                    )
                    masked_x = x[mask == 1]
                    masked_y = recon_t[mask == 1]
                    if masked_x.numel() == 0:
                        print("Skip batch: no masked temporal-spatial patches")
                        continue
                    lt = self.criterion(masked_y, masked_x)

                    lf = self._masked_mse_band2(recon_f, freq_target, mask_f)

                    la = self.model.alignment_loss(feats_t, feats_f)

                    loss = lt + getattr(self.params, 'lambda_f', 1.0) * lf + getattr(self.params, 'lambda_align', 0.1) * la
                else:
                    # No masking: still train both branches (reconstruct all)
                    recon_t, recon_f, feats_t, feats_f, _, freq_target = self.model.forward_pretrain(
                        x, mask_t=None, mask_f=None
                    )
                    lt = self.criterion(recon_t, x)
                    lf = self.criterion(recon_f, freq_target)
                    la = self.model.alignment_loss(feats_t, feats_f)
                    loss = lt + getattr(self.params, 'lambda_f', 1.0) * lf + getattr(self.params, 'lambda_align', 0.1) * la

                if not torch.isfinite(loss):
                    print("Skip batch: non-finite loss")
                    continue

                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.data.cpu().numpy())
                
            mean_loss = np.mean(losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.6f}')
            
            if mean_loss < best_loss:
                model_path = rf'{self.params.model_dir}/epoch{epoch+1}_loss{mean_loss}.pth'
                torch.save(self.model.state_dict(), model_path)
                print("model save in " + model_path)
                best_loss = mean_loss
