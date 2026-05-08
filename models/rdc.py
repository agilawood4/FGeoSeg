import torch, torch.nn as nn

try:
    from torchvision.ops import DeformConv2d
    HAS_DCN = True
except Exception:
    HAS_DCN = False

class RDC(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.use_dcn = HAS_DCN
        if self.use_dcn:
            self.offset = nn.Conv2d(in_ch, 2*kernel_size*kernel_size, kernel_size=3, padding=1, bias=True)
            self.modulator = nn.Conv2d(in_ch, kernel_size*kernel_size, kernel_size=3, padding=1, bias=True)
            self.dcn = DeformConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
            self.bn   = nn.BatchNorm2d(out_ch)
            self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_dcn:
            offset = self.offset(x)
            m = torch.sigmoid(self.modulator(x)) * 2
            out = self.dcn(x, offset, m)
            return out
        else:
            return self.act(self.bn(self.conv(x)))

def wrap_decoder_with_rdc(decoder, replace_stages=(0,)):
    try:
        blocks = getattr(decoder, 'blocks', None)
        if blocks is None:
            return decoder
        for i, blk in enumerate(blocks):
            if i in replace_stages:
                if hasattr(blk, 'conv1') and isinstance(blk.conv1, nn.Conv2d):
                    in_ch = blk.conv1.in_channels
                    out_ch= blk.conv1.out_channels
                    blk.conv1 = RDC(in_ch, out_ch, 3, 1, 1)
        return decoder
    except Exception:
        return decoder
