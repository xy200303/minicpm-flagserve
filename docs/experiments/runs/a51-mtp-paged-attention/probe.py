import torch
import ixformer


def reference(q, k, v, kv_len, scale):
    q_len = q.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    positions = torch.arange(q_len, device=q.device) + kv_len - q_len
    allowed = torch.arange(kv_len, device=q.device)[None, :] <= positions[:, None]
    scores = scores.masked_fill(~allowed[None, :, :], -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v.float()).to(q.dtype)


def main():
    torch.manual_seed(1234)
    device = "cuda"
    q_len = 4
    kv_len = 8
    q_heads = 16
    kv_heads = 2
    head_size = 128
    page_size = 16
    pages = torch.tensor([3, 1], dtype=torch.int32)
    scale = head_size**-0.5
    q = torch.randn(q_len, q_heads, head_size, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(6, page_size, kv_heads, head_size, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    output = torch.empty_like(q)
    block_tables = pages[None, :].to(device)
    context_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    cu_query_lens = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    ixformer.vllm_paged_attention_mtp(
        output, q,
        k_cache.permute(0, 2, 1, 3), v_cache.permute(0, 2, 1, 3),
        kv_heads, scale, block_tables, context_lens,
        page_size, kv_len,
        causal=True, max_query_len=q_len, cu_query_lens=cu_query_lens,
    )
    torch.cuda.synchronize()
    k = k_cache[pages.long()].reshape(-1, kv_heads, head_size)[:kv_len]
    v = v_cache[pages.long()].reshape(-1, kv_heads, head_size)[:kv_len]
    expected = reference(q, k, v, kv_len, scale)
    diff = (output.float() - expected.float()).abs()
    print("output", output.flatten()[:8].tolist())
    print("expected", expected.flatten()[:8].tolist())
    print("max_diff", diff.max().item(), "mean_diff", diff.mean().item())
    torch.testing.assert_close(output, expected, rtol=0.05, atol=0.05)


if __name__ == "__main__":
    main()
