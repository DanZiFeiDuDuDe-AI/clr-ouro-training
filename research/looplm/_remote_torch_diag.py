import os
import torch
import numpy as np
print('torch', torch.__version__)
print('env_device', repr(os.environ.get('V24_O3BK_DEVICE')))
print('cuda_available', torch.cuda.is_available())
print('cuda_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    x=torch.randn(1024,2048,device='cuda'); y=x@x.T; torch.cuda.synchronize(); print('cuda_ok',float(y.mean()))
import sys
sys.path.insert(0, r'D:/training')
import a1_v24_o3b_k as k
print('module_device_name', repr(k.DEVICE_NAME))
import time
from a1_v24_o3b import MRCache
import a1_v24_o3b_j as j
t0=time.perf_counter(); cache=MRCache(k.CACHE_ROOT); infos,pmap=j.build_infos(cache); token,_=j.load_token_arrays(); carriers=j.aligned_carriers(infos,token); H=carriers['J-I2b']['train']; y=infos['train']['entity'][:len(H)]
print('carrier_shape',H.shape,'classes',len(pmap),'load_s',time.perf_counter()-t0)
t1=time.perf_counter(); rec=j.multiclass_fit(H,y,{'train':(H,y)},len(pmap),torch.device('cuda'),steps=2,seed=42); print('one_fit_s',time.perf_counter()-t1,'acc',rec['train']['accuracy'])
R,BS,D,C=12,1024,256,2378
X=torch.randn(R,BS,D,device='cuda'); W=torch.randn(R,C,D,device='cuda',requires_grad=True); b=torch.randn(R,C,device='cuda',requires_grad=True); opt=torch.optim.AdamW([W,b],lr=3e-4)
torch.cuda.synchronize(); t2=time.perf_counter()
for _ in range(2):
    z=torch.bmm(X,W.transpose(1,2))+b[:,None,:]; loss=z.square().mean(); opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize(); print('batched_bmm_s',time.perf_counter()-t2)
Xe=torch.randn(R,4096,D,device='cuda'); torch.cuda.synchronize(); t3=time.perf_counter(); z=torch.bmm(Xe,W.transpose(1,2)); torch.cuda.synchronize(); print('batched_eval_s',time.perf_counter()-t3,float(z.mean()))
t4=time.perf_counter();
for shape,q in [((1024,2048),256),((2378,2048),64),((600,2048),64)]:
    A=np.random.default_rng(1).normal(size=shape).astype('float32'); _=k.top_right_dirs(A,q); print('rand_dirs',shape,q,time.perf_counter()-t4); t4=time.perf_counter()
