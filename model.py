import torch
from torch import nn


class SimpleConvSegHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # baseline head 只允许一个 1x1 conv，不引入 decoder 或额外非线性。
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


@torch.no_grad()
def update_ema(student, teacher, global_step, ema_weight=0.99):
    # global_step=0 时 alpha=0，可在 finetune reset 后让 teacher 首步对齐 student。
    alpha = min(1.0 - 1.0 / float(global_step + 1), ema_weight)
    for ema_p, p in zip(teacher.parameters(), student.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)
    for ema_b, b in zip(teacher.buffers(), student.buffers()):
        ema_b.copy_(b)
