import sys
import torch, transformers, bitsandbytes, accelerate
with open(r'D:\training\_env_probe4.txt', 'w') as f:
    f.write('torch ' + torch.__version__ + ' cuda ' + (torch.version.cuda or '') + ' avail ' + str(torch.cuda.is_available()) + chr(10))
    f.write('transformers ' + transformers.__version__ + chr(10))
    f.write('bnb ' + bitsandbytes.__version__ + chr(10))
    f.write('acc ' + accelerate.__version__ + chr(10))
