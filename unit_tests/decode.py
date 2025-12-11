import torch
import flashinfer
kv_len = 2048
num_qo_heads = 64
num_kv_heads = 8
head_dim = 128
q = torch.randn(num_qo_heads, head_dim).half().to("cuda:0")
k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
o = flashinfer.single_decode_with_kv_cache(q, k, v, use_tensor_cores=True)