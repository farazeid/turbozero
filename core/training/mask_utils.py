import jax
import jax.numpy as jnp


def compute_shapley_weights(n: int) -> jnp.ndarray:
    """
    Computes importance sampling weights for subset sizes k in [1, n-1].
    w_k = (n-1) / (k * (n-k))
    """
    ks = jnp.arange(1, n)
    weights = (n - 1) / (ks * (n - ks))
    # Normalize to probability distribution
    probs = weights / jnp.sum(weights)
    return probs


def sample_shapley_masks(
    key: jax.random.PRNGKey, batch_size: int, height: int, width: int
) -> jnp.ndarray:
    """
    Samples feature masks according to the Shapley distribution.

    Args:
        key: JAX PRNG key
        batch_size: Number of masks to sample
        height: Board height
        width: Board width

    Returns:
        masks: Binary masks of shape (batch_size, height, width, 1)
    """
    n = height * width

    # 1. Pre-calculate weights for k (subset size)
    # We can cache this if needed, but it's cheap for small N=361
    probs = compute_shapley_weights(n)

    # 2. Sample k for each batch element
    k_key, idx_key = jax.random.split(key)
    # values are 1..n-1, indices are 0..n-2
    k_indices = jax.random.choice(k_key, n - 1, shape=(batch_size,), p=probs)
    ks = k_indices + 1  # Convert index to actual size k

    # 3. Sample indices for each batch element
    # We want to select k indices out of n.
    # Approach: Generate random permutation of indices [0..n-1] and take first k.

    def sample_single_mask(key, k):
        # random permutation of 0..n-1
        perm = jax.random.permutation(key, n)
        # Select first k indices
        # We can create a mask directly
        # Indices are effectively selected if their rank in permutation is < k
        # But constructing array of size n with ones at perm[:k] is tricky with dynamic k in JIT?
        # Alternative: Generate uniform noise, find top-k?
        # "Gumbel-Top-K" logic or simply argsort of random noise.

        noise = jax.random.uniform(key, (n,))
        # argsort gives indices of sorted elements.
        # If we take indices of top-k values, that's a random subset.
        # Actually, simpler: just test if rank < k?
        # But we need exactly k ones.

        # Simpler:
        # 1. Generate random noise U ~ Uniform(0,1) for each pixel.
        # 2. Find the k-th largest value (k-th quantile).
        # 3. Set top-k to 1.

        # jnp.argsort is valid in JIT
        # To get binary mask:
        # Create a boolean array where indices in top-k are True.

        # idxs = jnp.argsort(noise) # Ascending
        # # top-k are the last k elements
        # # selected_idxs = idxs[n-k:]
        # This involves dynamic slice which is tricky in JIT unless we use mask.

        # Better:
        # mask = (rank(noise) >= n-k)
        # rank can be got by argsort(argsort(noise))

        ranks = jnp.argsort(jnp.argsort(noise))  # 0 to n-1
        mask_flat = ranks >= (n - k)
        return mask_flat.reshape((height, width, 1)).astype(jnp.float32)

    # Vectorize over batch
    keys = jax.random.split(idx_key, batch_size)
    masks = jax.vmap(sample_single_mask)(keys, ks)

    return masks
