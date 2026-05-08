import os, yaml, torch
from models.student_segformer_b0 import SegFormerB0Sky

class Wrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m=m
    def forward(self, x):
        out = self.m(x)
        logits = out.logits if hasattr(out,'logits') else out
        return logits

def main():
    cfg = yaml.safe_load(open('configs/config.yaml','r'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SegFormerB0Sky(num_labels=cfg['num_classes'])
    ckpt = os.path.join(cfg['save_dir'], 'student_segformer_b0_best.pth')
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
    model.eval().to(device)

    dummy = torch.randn(1,3,cfg['img_size'],cfg['img_size'], device=device)
    onnx_path = cfg['export']['onnx_path']
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(Wrapper(model), dummy, onnx_path,
                      input_names=['images'], output_names=['logits'],
                      opset_version=int(cfg['export']['opset']),
                      dynamic_axes={'images':{0:'B',2:'H',3:'W'}, 'logits':{0:'B',2:'H',3:'W'}})
    print("Saved ONNX to", onnx_path)

if __name__=='__main__':
    main()
