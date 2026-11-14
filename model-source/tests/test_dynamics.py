import unittest
import torch
from bdh_nca import BDHNCA, Config
from bdh_nca.model import shift, TissueState
from bdh_nca.data import make_example, dataset_audit
from bdh_nca.evaluate import controls

class DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_parameter_budget(self):
        for width in (32,96,256):
            n=BDHNCA(Config(hidden=width)).parameter_count
            self.assertGreaterEqual(n,5000);self.assertLessEqual(n,50000)

    def test_no_wrap_or_instant_nonlocal_signal(self):
        x=torch.zeros(1,1,12,12);x[0,0,4,0]=1
        self.assertEqual(float(shift(x,0,1)[0,0,4,-1]),0)
        m=BDHNCA();a=torch.zeros(1,24,16,16);a[:,0]=1
        s=m.initial(a);s.cells[:,4,8,8]=1
        out=m(s,fire=torch.ones(1,1,16,16))
        # Zero update initialization: only local voltage transport on the first tick.
        self.assertEqual(float(out.cells[0,4,8,2].detach()),0)
        self.assertGreater(float(out.cells[0,4,8,7].detach()),0)

    def test_long_horizon_bounds_and_reset(self):
        m=BDHNCA();x=torch.ones(1,24,16,16)*.8;x[:,22:]=0
        s=m.initial(x)
        with torch.no_grad():
            for i in range(600):
                u=torch.zeros(1,2,16,16);u[:,0]=1
                s=m(s,u,torch.ones(1,1,16,16))
                self.assertLessEqual(float(s.cells.abs().max()),1.000001)
                self.assertLessEqual(float((s.edge_fast+s.edge_slow).max()),1.000001)
                self.assertTrue(torch.isfinite(s.cells).all())
        self.assertGreater(float(s.edge_slow.mean()),0)
        empty=m.initial(torch.zeros_like(x))
        dirty=TissueState(empty.cells,s.edge_fast,s.edge_slow,s.delayed)
        with torch.no_grad():out=m(dirty)
        self.assertEqual(sum(float(t.abs().sum()) for t in out.tensors()),0)

    def test_ablations_remove_memory(self):
        x=make_example(24,0,"planaria").seed
        for c in (Config(plasticity=False),Config(consolidation=False),Config(long_range=False)):
            m=BDHNCA(c);s=m.initial(x)
            with torch.no_grad():
                for _ in range(10):s=m(s)
            if not c.plasticity:self.assertEqual(float(s.edge_fast.sum()+s.edge_slow.sum()),0)
            if not c.consolidation:self.assertEqual(float(s.edge_slow.sum()+s.cells[:,21].abs().sum()),0)
            if not c.long_range:self.assertEqual(float(s.delayed.abs().sum()+s.edge_fast[:,4:].sum()),0)

    def test_normal_data_and_intervention_separation(self):
        self.assertEqual(dataset_audit(128,32)["two_head_targets"],0)
        e=make_example(32,0,"planaria")
        for name in ("tail_pulse","ectopic_pulse"):
            c=controls(e.seed,name)
            self.assertEqual(c.shape[1],2)
            self.assertLessEqual(float(c.abs().max()),1)
        self.assertFalse(torch.equal(make_example(32,0,split="train").seed,make_example(32,0,split="test").seed))

    def test_gradients_reach_cell_and_edge_routes(self):
        torch.manual_seed(1);m=BDHNCA();torch.nn.init.normal_(m.update[-1].weight,std=.02)
        state=m.initial(make_example(24,0,"planaria").seed)
        for _ in range(4):state=m(state,fire=torch.ones(1,1,24,24))
        state.cells[:,:20].square().mean().backward()
        for name in ("query.weight","value.weight","fast_factor.weight","update.0.weight"):
            g=dict(m.named_parameters())[name].grad
            self.assertIsNotNone(g);self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0)

    def test_zero_fate_can_recover_after_negative_proposal(self):
        m=BDHNCA()
        with torch.no_grad():m.update[-1].bias[2]=-.1
        state=m.initial(make_example(24,0,"planaria").seed)
        result=m(state,fire=torch.ones(1,1,24,24))
        self.assertEqual(float(result.cells[:,2].detach().abs().max()),0)
        (result.cells[:,2]-1).square().mean().backward()
        self.assertLess(float(m.update[-1].bias.grad[2]),0)

    def test_edge_history_and_consolidation_change_voltage(self):
        m=BDHNCA();x=torch.zeros(1,24,16,16);x[:,0]=1
        x[0,4,8,7]=1;x[0,4,8,9]=-1
        base=m.initial(x.clone());reinforced=m.initial(x.clone())
        reinforced.edge_slow[0,2,8,8]=.8
        mask=torch.ones(1,1,16,16)
        with torch.no_grad():
            a=m(base,fire=mask);b=m(reinforced,fire=mask)
        self.assertGreater(float(b.cells[0,4,8,8]),float(a.cells[0,4,8,8]))
        recalled=m.initial(x.clone());recalled.cells[0,21,8,8]=.8
        with torch.no_grad():r=m(recalled,fire=mask)
        self.assertGreater(float(r.cells[0,4,8,8]),float(a.cells[0,4,8,8]))

    def test_damage_clears_incident_but_preserves_remote_history(self):
        m=BDHNCA();s=m.initial(torch.ones(1,24,20,20))
        s.edge_fast.fill_(.3);s.edge_slow.fill_(.5);s.delayed.fill_(.4)
        keep=torch.ones(1,1,20,20);keep[:,:,10,10]=0
        d=m.damage(s,keep)
        self.assertEqual(float(d.cells[:,:,10,10].sum()),0)
        self.assertEqual(float(d.edge_slow[0,2,10,11]),0)
        self.assertEqual(float(d.delayed[0,0,:,10,16].sum()),0)
        self.assertAlmostEqual(float(d.edge_slow[0,0,3,3]),.5)

if __name__=="__main__":unittest.main()
