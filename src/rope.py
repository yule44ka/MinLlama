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
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    batch_size, seqlen, n_local_heads, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.

    # Generate sinusoidal and cosine frequencies
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float)
    log_freqs = torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / float(head_dim)
    inv_freqs = 1.0 / (theta ** log_freqs)

    # Compute the angles for each position and each dimension
    sinusoid_inp = positions.unsqueeze(1) * inv_freqs.unsqueeze(0)
    sin_inp, cos_inp = torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

    # Expand the sin and cos tensors to match the query and key tensor shapes
    sin_inp = sin_inp.unsqueeze(0).unsqueeze(0).expand(batch_size, seqlen, n_local_heads, -1)
    cos_inp = cos_inp.unsqueeze(0).unsqueeze(0).expand(batch_size, seqlen, n_local_heads, -1)

    # Expand sin_inp and cos_inp to match the query and key tensor shapes
    sin_inp = sin_inp.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, seqlen, n_local_heads, head_dim // 2)
    cos_inp = cos_inp.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, max_seq_len, n_local_heads,
                                                                     head_dim // 2)

    # Apply rotary embeddings
    query_emb_real = query_real * cos_inp - query_imag * sin_inp
    query_emb_imag = query_real * sin_inp + query_imag * cos_inp
    key_emb_real = key_real * cos_inp - key_imag * sin_inp
    key_emb_imag = key_real * sin_inp + key_imag * cos_inp

    # Combine the real and imaginary parts
    query_emb = torch.stack([query_emb_real, query_emb_imag], dim=-1).reshape(query.shape)
    key_emb = torch.stack([key_emb_real, key_emb_imag], dim=-1).reshape(key.shape)

    return query_emb, key_emb



    # freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    # t = torch.arange(max_seq_len, device=device)
    # freqs = torch.outer(t, freqs).float()
    #
    # freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    #
    # query_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    # key_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    #
    # freqs_cis = reshape_for_broadcast(freqs_cis, query_)
    #
    # query_out = torch.view_as_real(query_ * freqs_cis).flatten(3).type_as(query)
    # key_out = torch.view_as_real(key_ * freqs_cis).flatten(3).type_as(key)
    #
    # # Return the rotary position embeddings for the query and key tensors
    # return query_out, key_out
