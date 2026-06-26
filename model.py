import torch
from torch import nn


class SimpleConvSegHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # baseline head 只允许一个 1x1 conv，不引入 decoder 或额外非线性。
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class ContextResidualHead(nn.Module):
    def __init__(self, in_channels, hidden=64, gamma_init=0.0):
        super().__init__()
        self.base = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.reduce = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
            bias=True,
        )
        self.act = nn.GELU()
        self.out = nn.Conv2d(hidden, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        base_logits = self.base(x)
        z = self.reduce(x)
        z = self.dw(z)
        z = self.act(z)
        res_logits = self.out(z)
        return base_logits + self.gamma * res_logits


def build_seg_head(in_channels, cfg):
    head_type = getattr(cfg, "HEAD_TYPE", "simple")
    if head_type == "simple":
        return SimpleConvSegHead(in_channels)
    if head_type == "context_residual":
        hidden = int(getattr(cfg, "CONTEXT_HEAD_HIDDEN", 64))
        gamma_init = float(getattr(cfg, "CONTEXT_HEAD_GAMMA_INIT", 0.0))
        return ContextResidualHead(in_channels, hidden=hidden, gamma_init=gamma_init)
    raise ValueError(f"Unknown HEAD_TYPE: {head_type}")


@torch.no_grad()
def update_ema(student, teacher, global_step, ema_weight=0.99):
    # global_step=0 时 alpha=0，可在 finetune reset 后让 teacher 首步对齐 student。
    alpha = min(1.0 - 1.0 / float(global_step + 1), ema_weight)
    for ema_p, p in zip(teacher.parameters(), student.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)
    for ema_b, b in zip(teacher.buffers(), student.buffers()):
        ema_b.copy_(b)
