"""Tiny exact-arithmetic CUDA checks; exits without keeping a GPU process."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from pact_eval.metrics import profile_counter

torch.manual_seed(7)
a=torch.randn(2,3,device='cuda'); b=torch.randn(3,4,device='cuda')
expected=a@b
counter=profile_counter()
with counter:
    with torch.inference_mode():
        actual=a@b
assert torch.equal(actual,expected)
assert counter.get_total_flops()==2*2*3*4
q=torch.randn(1,2,8,16,device='cuda',dtype=torch.float16)
expected=F.scaled_dot_product_attention(q,q,q)
counter=profile_counter()
with counter:
    with torch.inference_mode():
        actual=F.scaled_dot_product_attention(q,q,q)
assert torch.equal(actual,expected)
assert counter.get_total_flops()==4*1*2*8*8*16,counter.get_flop_counts()
print('FLOP_COUNTER_CUDA_PASS',torch.__version__,counter.get_total_flops())
