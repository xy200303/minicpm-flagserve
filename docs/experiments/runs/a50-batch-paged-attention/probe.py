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
    batch = 1
    q_len = 1
    kv_len = 8
    q_heads = 16
    kv_heads = 2
    head_size = 128
    page_size = 16
    pages = torch.tensor([3, 1], dtype=torch.int32)
    max_pages = 6
    scale = head_size**-0.5
    q = torch.randn(q_len, q_heads, head_size, device=device, dtype=torch.float16)
    k_cache = torch.randn(6 * page_size, kv_heads, head_size, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    output = torch.empty_like(q)
    lse = torch.empty(0, 0, device=device, dtype=torch.float32)
    q_index = torch.tensor([0, q_len], dtype=torch.int32)
    kv_index = torch.tensor([0, kv_len], dtype=torch.int32)
    block_tables = torch.zeros((batch, max_pages), dtype=torch.int32)
    block_tables[0, :pages.numel()] = pages
    cpu_block_tables = block_tables.cpu()
    ixformer.batch_paged_attention(
        output, q, (k_cache, v_cache), lse,
        q_index.to(device), kv_index.to(device), block_tables.to(device),
        q_index, kv_index, cpu_block_tables,
        batch, max_pages, page_size, kv_heads, scale, False,
    )
    torch.cuda.synchronize()
    k = k_cache.view(-1, page_size, kv_heads, head_size)[pages.long()].reshape(-1, kv_heads, head_size)[:kv_len]
    v = v_cache.view(-1, page_size, kv_heads, head_size)[pages.long()].reshape(-1, kv_heads, head_size)[:kv_len]
    expected = reference(q, k, v, kv_len, scale)
    diff = (output.float() - expected.float()).abs()
    print("output", output.flatten()[:8].tolist())
    print("expected", expected.flatten()[:8].tolist())
    print("max_diff", diff.max().item(), "mean_diff", diff.mean().item())
    print("lse", lse)
    torch.testing.assert_close(output, expected, rtol=0.05, atol=0.05)


if __name__ == "__main__":
    main()
