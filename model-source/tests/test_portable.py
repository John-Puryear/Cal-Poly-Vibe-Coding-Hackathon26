import unittest
import numpy as np
import torch
from bdh_nca.model import BDHNCA,Config
from bdh_nca.numpy_runtime import step

class PortableTests(unittest.TestCase):
    def test_independent_recurrent_parity_with_ablations(self):
        torch.set_num_threads(2);torch.manual_seed(123)
        configs=[Config(hidden=32),Config(hidden=32,plasticity=False),
                 Config(hidden=32,consolidation=False),Config(hidden=32,long_range=False),
                 Config(hidden=32,interaction=False,long_radius=3)]
        from dataclasses import asdict
        for c in configs:
            m=BDHNCA(c)
            torch.nn.init.normal_(m.update[-1].weight,std=.04)
            x=torch.rand(2,24,16,19)*2-1;x[:,0]=1;x[:,:,3:5,7:9]=0
            s=m.initial(x);s.edge_fast.fill_(.2);s.edge_slow.fill_(.3)
            ns=tuple(t.detach().numpy().copy() for t in s.tensors())
            weights={k:v.detach().numpy() for k,v in m.state_dict().items()}
            with torch.no_grad():
                for i in range(12):
                    mask=(torch.rand_like(x[:,:1])<.5).float()
                    control=torch.rand(2,2,16,19);control[:,0]=control[:,0]*2-1
                    s=m(s,control,mask)
                    ns=step(ns,control.numpy(),mask.numpy(),weights,asdict(c))
                    for a,b in zip(s.tensors(),ns):
                        np.testing.assert_allclose(a.numpy(),b,atol=1e-5,rtol=1e-5)

if __name__=="__main__":unittest.main()
