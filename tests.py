import unittest
import torch
from autoencoder import AutoEncoder



class TestGradientAdjustment(unittest.TestCase):
    def test_gradient_adjustment(self):
        # Create a toy model with a simple weight matrix
        
        # compute forward pass
        # Original einsum implementation
        model1 = AutoEncoder.default()
        model2 = AutoEncoder.default()

        model1.W_dec = model2.W_dec
        grad = torch.randn_like(model1.W_dec)
        model1.W_dec.grad = grad
        model2.W_dec.grad = grad

        model1.remove_parallel_component_einsum()
        model2.remove_parallel_component()
        self.assertTrue(torch.allclose(model1.W_dec.grad, model2.W_dec.grad, atol=1e-6))



if __name__ == '__main__':
    unittest.main()