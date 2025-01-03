import torch

import unittest

from apex.contrib.multihead_attn import SelfMultiheadAttn

class SelfMultiheadAttnNormAddTest(unittest.TestCase):
    def setUp(self, seed=1234):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.seq_length   = 80
        self.sequences    = 10
        self.hidden_dim   = 1024
        self.heads        = 16
        self.dropout_prob = 0.0

        self.ref_layer = SelfMultiheadAttn(self.hidden_dim, 
                                           self.heads, 
                                           dropout=self.dropout_prob, 
                                           bias=False, 
                                           include_norm_add=True, 
                                           impl='default')
        self.ref_layer.cuda().half()
        self.ref_layer.reset_parameters()
        self.ref_inputs = torch.randn(self.seq_length, self.sequences, self.hidden_dim, 
                                      dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

        # Reset seed so parameters are identical
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.tst_layer = SelfMultiheadAttn(self.hidden_dim, 
                                           self.heads, 
                                           dropout=self.dropout_prob, 
                                           bias=False, 
                                           include_norm_add=True, 
                                           impl='fast')
        self.tst_layer.cuda().half()
        self.tst_layer.reset_parameters()
        
        self.tst_inputs = torch.randn(self.seq_length, self.sequences, self.hidden_dim, 
                                      dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

    def test_self_multihead_attn_norm_add(self) :
        grads         = torch.randn_like(self.tst_inputs)

        ref_outputs,_ = self.ref_layer.forward(self.ref_inputs, self.ref_inputs,, self.ref_inputs

        tst_outputs,_ = self.tst_layer.forward(self.tst_inputs, self.tst_inputs,, self.tst_inputs
        
        self.ref_inputs.backward(grads)
        self.tst_inputs.backward(grads)

        self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,  atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad, atol=1e-3, rtol=1e-3))

if __name__ == '__main__':
    unittest.main()
