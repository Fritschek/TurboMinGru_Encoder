import torch
import torch.nn.functional as F
import fused_parallel_scan  # the compiled extension

# Fused version using your custom CUDA kernel.
def fused_parallel_scan_fn(log_coeffs, log_values):
    return fused_parallel_scan.fused_parallel_scan_cuda(log_coeffs, log_values)

# Reference version: the original implementation.
def reference_parallel_scan(log_coeffs, log_values):
    # log_coeffs: (batch, T, hidden)
    # log_values: (batch, T+1, hidden)
    a_star = torch.cumsum(log_coeffs, dim=1)
    # Pad along the time dimension at the beginning.
    a_star = F.pad(a_star, (0, 0, 1, 0))  # shape: (batch, T+1, hidden)
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    # Drop the first time step so the output has shape (batch, T, hidden)
    return torch.exp(log_h)[:, 1:]

def main():
    # Use a fixed seed for reproducibility.
    torch.manual_seed(42)
    
    # Define dimensions.
    batch = 20
    T = 100
    hidden = 40

    # Create random inputs on CUDA.
    log_coeffs = torch.randn(batch, T, hidden, device='cuda')
    log_values = torch.randn(batch, T+1, hidden, device='cuda')
    
    # Compute outputs.
    out_fused = fused_parallel_scan_fn(log_coeffs, log_values)
    out_ref   = reference_parallel_scan(log_coeffs, log_values)
    
    # Print results.
    print("Fused output:")
    print(out_fused)
    print("\nReference output:")
    print(out_ref)
    
    # Compare using allclose with a tolerance.
    if torch.allclose(out_fused, out_ref, atol=1e-5, rtol=1e-4):
        print("\nOutputs are close!")
    else:
        diff = torch.abs(out_fused - out_ref).max()
        print("\nMaximum absolute difference:", diff.item())

if __name__ == '__main__':
    main()
