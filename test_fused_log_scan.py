import torch
import torch.nn.functional as F
import fused_log_scan  # Your compiled extension with fused_log_scan_cuda(...)
import time

def reference_parallel_scan_log_3d(log_coeffs_3d, log_values_3d):
    """
    Reference (unfused) implementation on 3D data.
    
    log_coeffs_3d: (batch, seq_len, hidden)
    log_values_3d: (batch, seq_len+1, hidden)
    
    Compute:
      a_star = pad(cumsum(log_coeffs_3d), left-pad along seq_len dimension)
      out = exp( a_star + logcumsumexp(log_values_3d - a_star) )[:, 1:, :]
      
    Since we’re working with 3D tensors, we’ll use torch.cat rather than F.pad.
    """
    batch, seq_len, hidden = log_coeffs_3d.shape
    csum = torch.cumsum(log_coeffs_3d, dim=1)  # shape: (batch, seq_len, hidden)
    zero_slice = torch.zeros(batch, 1, hidden, device=csum.device, dtype=csum.dtype)
    a_star = torch.cat([zero_slice, csum], dim=1)  # shape: (batch, seq_len+1, hidden)
    b_vals = log_values_3d - a_star
    log_h0_plus_b_star = torch.logcumsumexp(b_vals, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:, :]  # drop the first time-step

def reference_parallel_scan_log_2d(log_coeffs_2d, log_values_2d):
    """
    2D reference implementation for flattened data.
    
    log_coeffs_2d: (B, seq_len)
    log_values_2d: (B, seq_len+1)
    
    Here we pad the time dimension on the left (i.e. (1,0)) so that:
      a_star[:,0] = 0 and a_star[:,t+1] = cumsum(log_coeffs_2d) for t=0,...,seq_len-1.
    """
    B, seq_len = log_coeffs_2d.shape
    # Pad on the left (i.e., add a zero column at index 0)
    a_star = F.pad(torch.cumsum(log_coeffs_2d, dim=1), (1, 0), value=0.0)  # shape: (B, seq_len+1)
    b_vals = log_values_2d - a_star
    log_h0_plus_b_star = torch.logcumsumexp(b_vals, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]  # shape: (B, seq_len)

def fused_parallel_scan_3d(log_coeffs_3d, log_values_3d):
    """
    Calls the fused kernel on flattened 2D data and then reshapes the result back to 3D.
    
    log_coeffs_3d: (batch, seq_len, hidden)
    log_values_3d: (batch, seq_len+1, hidden)
    """
    batch, seq_len, hidden = log_coeffs_3d.shape
    B = batch * hidden
    # Flatten so that each row corresponds to a (seq_len) scan for one (batch, hidden) pair.
    log_coeffs_2d = log_coeffs_3d.permute(0,2,1).contiguous().view(B, seq_len)
    log_values_2d = log_values_3d.permute(0,2,1).contiguous().view(B, seq_len+1)
    
    # Call the fused kernel; it expects (B, seq_len) and (B, seq_len+1)
    out_2d = fused_log_scan.fused_log_scan_cuda(log_coeffs_2d, log_values_2d, seq_len)
    
    # Reshape back to 3D: (B, seq_len) -> (batch, hidden, seq_len) then permute to (batch, seq_len, hidden)
    out_3d = out_2d.view(batch, hidden, seq_len).permute(0,2,1).contiguous()
    return out_3d

def test_fused_vs_reference_3d(num_runs=50, batch_size=32, seq_len=1000, hidden_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Generate random 3D data mimicking the training branch
    torch.manual_seed(0)
    log_coeffs_3d = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype) * 0.01
    log_values_3d = torch.randn(batch_size, seq_len+1, hidden_size, device=device, dtype=dtype) * 0.01

    # Warm-up
    for _ in range(5):
        ref_out = reference_parallel_scan_log_3d(log_coeffs_3d, log_values_3d)
        fus_out = fused_parallel_scan_3d(log_coeffs_3d, log_values_3d)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Correctness check
    ref_out = reference_parallel_scan_log_3d(log_coeffs_3d, log_values_3d)
    fus_out = fused_parallel_scan_3d(log_coeffs_3d, log_values_3d)
    max_diff = (ref_out - fus_out).abs().max().item()
    close_ok = torch.allclose(ref_out, fus_out, atol=1e-5, rtol=1e-5)
    print(f"Correctness check: allclose={close_ok}, max_diff={max_diff:.3e}")

    # Speed measurement
    def time_method(fn, n):
        start = time.time()
        for _ in range(n):
            fn(log_coeffs_3d, log_values_3d)
        if device == 'cuda':
            torch.cuda.synchronize()
        return (time.time() - start) / n

    t_ref = time_method(reference_parallel_scan_log_3d, num_runs)
    t_fused = time_method(fused_parallel_scan_3d, num_runs)
    print(f"\nTimings (batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}):")
    print(f"  Reference: {t_ref*1000:.3f} ms/run")
    print(f"  Fused:     {t_fused*1000:.3f} ms/run")
    speedup = t_ref / t_fused if t_fused > 0 else float('inf')
    print(f"  => Speedup: {speedup:.2f}x\n")

if __name__ == "__main__":
    test_fused_vs_reference_3d(num_runs=50, batch_size=32, seq_len=1100, hidden_size=128)
