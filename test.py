import torch
from mamba_ssm import Mamba
from mamba_ssm import Mamba2

if __name__ == "__main__":
    print("Running mamba!")

    batch, length, dim = 2, 64, 512
    x = torch.randn(batch, length, dim).to("cuda")
    model = Mamba2(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")
    y = model(x)
    assert y.shape == x.shape
    print(x, "-->", y)
