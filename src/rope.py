from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    _, seqlen, _, _ = query.shape
    device = query.device
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    thetas = theta ** (-2.0 * (torch.arange(0, head_dim // 2)) / head_dim)
    m_thetas = torch.outer(torch.arange(seqlen), thetas)
    cos_thetas = reshape_for_broadcast(torch.cos(m_thetas), query_real)
    sin_thetas = reshape_for_broadcast(torch.sin(m_thetas), query_real)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.

    # raise NotImplementedError

    query_out = torch.zeros_like(query)
    query_out[..., ::2] = query_real * cos_thetas - query_imag * sin_thetas
    query_out[..., 1::2] = query_real * sin_thetas + query_imag * cos_thetas

    key_out = torch.zeros_like(key)
    key_out[..., ::2] = key_real * cos_thetas - key_imag * sin_thetas
    key_out[..., 1::2] = key_real * sin_thetas + key_imag * cos_thetas
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out
