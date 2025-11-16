from flash_attn_interface import flash_attn_func

import torch

try:

  @torch.library.custom_op("flash_attention::flash_attn", mutates_args=())
  def flash_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # dropout will be 0 for inference
    dropout_p: float = 0.0,
    causal: bool = False,
  ) -> torch.Tensor:
    return flash_attn_func(q, k, v, causal=causal)

  @flash_attn_wrapper.register_fake
  def flash_attn_fake(q, k, v, dropout_p=0.0, causal=False):
    # Output shape is the same as q
    return q.new_empty(q.shape)
except AttributeError as error:
  FLASH_ATTN_ERROR = error

  def flash_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
  ) -> torch.Tensor:
    assert False, f"Could not define flash_attn_wrapper: {FLASH_ATTN_ERROR}"


if __name__ == "__main__":
  device = torch.device("cuda")
  dtype = torch.float16

  # Test 1: Basic attention test
  q = torch.randn(
    2, 8, 4, 64, device=device, dtype=dtype
  )  # (batch, seq_len, heads, head_dim)
  k = torch.randn(2, 8, 4, 64, device=device, dtype=dtype)
  v = torch.randn(2, 8, 4, 64, device=device, dtype=dtype)

  out = flash_attn_wrapper(q, k, v, dropout_p=0.0, causal=False)
  assert out.shape == (2, 8, 4, 64)

  # Test 2: Causal attention test
  out_causal = flash_attn_wrapper(q, k, v, dropout_p=0.0, causal=True)
  assert out_causal.shape == (2, 8, 4, 64)

  # Test 3: Different sequence length
  q2 = torch.randn(1, 16, 8, 128, device=device, dtype=dtype)
  k2 = torch.randn(1, 16, 8, 128, device=device, dtype=dtype)
  v2 = torch.randn(1, 16, 8, 128, device=device, dtype=dtype)

  out2 = flash_attn_wrapper(q2, k2, v2, dropout_p=0.0, causal=False)
  assert out2.shape == (1, 16, 8, 128)
