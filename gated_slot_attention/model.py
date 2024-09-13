from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from zeta.nn.modules.glu import GLU
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm
from zeta import OutputHead


class GatedSlotAttention(nn.Module):
    """
    Gated Slot Attention (GSA) layer implementation.

    This module implements the Gated Slot Attention mechanism as described in the provided document.
    It incorporates a gating mechanism to enable forgetting of historical information and introduces
    a recency inductive bias.

    Attributes:
        dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        d_head (int): Dimensionality of each attention head.
        m (int): Number of memory slots.
        tau (float): Damping factor for the forget gate.
        W_q (nn.Linear): Linear projection for queries.
        W_k (nn.Linear): Linear projection for keys.
        W_v (nn.Linear): Linear projection for values.
        W_alpha (nn.Linear): Linear projection for forget gates.
        W_o (nn.Linear): Linear projection for output.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        m: int,
        tau: float = 0.1,
    ):
        """
        Initialize the GatedSlotAttention module.

        Args:
            dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            m (int): Number of memory slots.
            tau (float): Damping factor for the forget gate.
        """
        super(GatedSlotAttention, self).__init__()
        assert (
            dim % num_heads == 0
        ), "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads  # Dimension per head
        self.m = m
        self.tau = tau

        # Linear projections for queries, keys, values, and forget gates
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_alpha = nn.Linear(dim, num_heads * m)

        # Output projection
        self.W_o = nn.Linear(num_heads * self.d_head, dim)

        # Activation functions
        self.act_fn = nn.SiLU()  # Swish activation function

        # Initialize memory slots Ke_t and Ve_t
        # These will be registered as buffers so they are not treated as parameters
        self.register_buffer(
            "Ke_t_init",
            torch.zeros(self.num_heads, self.m, self.d_head),
        )
        self.register_buffer(
            "Ve_t_init",
            torch.zeros(self.num_heads, self.m, self.d_head),
        )

    def forward(
        self, x: TensorType["batch_size", "seq_len", "dim"]
    ) -> TensorType["batch_size", "seq_len", "dim"]:
        """
        Forward pass of the Gated Slot Attention layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, _ = x.size()

        # Compute queries, keys, values
        # Shapes: (batch_size, seq_len, num_heads * d_head)
        q = self.act_fn(self.W_q(x))
        k = self.act_fn(self.W_k(x))
        v = self.act_fn(self.W_v(x))

        # Reshape to (batch_size, seq_len, num_heads, d_head)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head)

        # Compute forget gates α_t
        # Shape: (batch_size, seq_len, num_heads * m)
        alpha = self.W_alpha(x)
        alpha = torch.sigmoid(alpha) ** (1 / self.tau)
        # Reshape to (batch_size, seq_len, num_heads, m)
        alpha = alpha.view(
            batch_size, seq_len, self.num_heads, self.m
        )

        # Initialize Ke_t and Ve_t for the batch
        # Shapes: (batch_size, num_heads, m, d_head)
        Ke_t = (
            self.Ke_t_init.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .clone()
        )
        Ve_t = (
            self.Ve_t_init.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .clone()
        )

        # Output tensor
        outputs = []

        for t in range(seq_len):
            # Get the t-th time step inputs
            # Shapes:
            # q_t: (batch_size, num_heads, d_head)
            # k_t: (batch_size, num_heads, d_head)
            # v_t: (batch_size, num_heads, d_head)
            # alpha_t: (batch_size, num_heads, m)
            q_t = q[:, t]  # (batch_size, num_heads, d_head)
            k_t = k[:, t]
            v_t = v[:, t]
            alpha_t = alpha[:, t]  # (batch_size, num_heads, m)

            # Expand dimensions for broadcasting
            # k_t_expanded: (batch_size, num_heads, 1, d_head)
            # v_t_expanded: (batch_size, num_heads, 1, d_head)
            # alpha_t_expanded: (batch_size, num_heads, m, 1)
            k_t_expanded = k_t.unsqueeze(
                2
            )  # (batch_size, num_heads, 1, d_head)
            v_t_expanded = v_t.unsqueeze(2)
            alpha_t_expanded = alpha_t.unsqueeze(
                -1
            )  # (batch_size, num_heads, m, 1)

            # Update Ke_t and Ve_t
            # Ke_t: (batch_size, num_heads, m, d_head)
            # Ve_t: (batch_size, num_heads, m, d_head)
            Ke_t = (
                alpha_t_expanded * Ke_t
                + (1 - alpha_t_expanded) * k_t_expanded
            )
            Ve_t = (
                alpha_t_expanded * Ve_t
                + (1 - alpha_t_expanded) * v_t_expanded
            )

            # Compute attention weights
            # q_t: (batch_size, num_heads, d_head, 1)
            q_t_expanded = q_t.unsqueeze(
                -1
            )  # (batch_size, num_heads, d_head, 1)
            # energy: (batch_size, num_heads, m)
            energy = torch.matmul(Ke_t, q_t_expanded).squeeze(
                -1
            )  # (batch_size, num_heads, m)
            # Apply softmax over memory slots dimension (m)
            attn_weights = F.softmax(
                energy, dim=-1
            )  # (batch_size, num_heads, m)

            # Compute output
            # attn_weights: (batch_size, num_heads, m, 1)
            attn_weights_expanded = attn_weights.unsqueeze(-1)
            # o_t: (batch_size, num_heads, d_head)
            o_t = torch.sum(
                attn_weights_expanded * Ve_t, dim=2
            )  # Sum over memory slots (m)

            # Collect outputs
            outputs.append(o_t)

        # Stack outputs along the sequence dimension
        # o: (batch_size, seq_len, num_heads, d_head)
        o = torch.stack(outputs, dim=1)

        # Reshape to (batch_size, seq_len, num_heads * d_head)
        o = o.contiguous().view(
            batch_size, seq_len, self.num_heads * self.d_head
        )

        # Output projection
        # y: (batch_size, seq_len, dim)
        y = self.W_o(o)

        return y


# # Example usage
# batch_size = 32
# seq_len = 128
# dim = 512
# num_heads = 4
# m = 64  # Number of memory slots
# tau = 0.1

# x = torch.randn(batch_size, seq_len, dim)
# gsa_layer = GatedSlotAttention(dim=dim, num_heads=num_heads, m=m, tau=tau)
# y = gsa_layer(x)  # y has shape (batch_size, seq_len, dim)
# print(y.shape)  # Should output: torch.Size([32, 128, 512])


class GSATransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        m: int,
        tau: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.m = m
        self.tau = tau
        self.attention = GatedSlotAttention(dim, heads, m, tau)

        # Norm
        self.norm = SimpleRMSNorm(dim)

        # GLU
        self.glu = GLU(
            dim,
            dim,
            nn.SiLU(),
            mult_bias=True,
        )

    def forward(
        self, x: TensorType["batch_size", "seq_len", "dim"]
    ) -> TensorType["batch_size", "seq_len", "dim"]:
        b, s, d = x.shape

        # residual
        residual = x

        # Attention
        x = self.attention(self.norm(x))

        x += residual

        # 2nd path
        residual_two = x

        # Norm
        x = self.norm(self.glu(x))

        return x + residual_two


class GSATransformer(nn.Module):
    """
    Gated Slot Attention Transformer model.

    This model applies multiple layers of Gated Slot Attention followed by an output head.

    Attributes:
        dim (int): Dimensionality of the model.
        heads (int): Number of attention heads.
        m (int): Number of memory slots.
        tau (float): Damping factor for the forget gate.
        depth (int): Number of transformer layers.
        vocab_size (int): Size of the vocabulary.
        max_seq_len (int): Maximum sequence length.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        m: int,
        tau: float = 0.1,
        depth: int = 1,
        vocab_size: int = 10000,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.m = m
        self.tau = tau
        self.depth = depth
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        logger.info(
            f"Initializing GSATransformer with dim={dim}, heads={heads}, m={m}, tau={tau}, depth={depth}, vocab_size={vocab_size}, max_seq_len={max_seq_len}"
        )

        self.embed = nn.Embedding(vocab_size, dim)
        logger.debug(
            f"Embedding layer initialized with vocab_size={vocab_size} and dim={dim}"
        )

        # Layers
        self.layers = nn.ModuleList(
            [
                GSATransformerBlock(dim, heads, m, tau)
                for _ in range(depth)
            ]
        )
        logger.debug(
            f"Initialized {depth} GSATransformerBlock layers"
        )

        # Norm
        self.norm = SimpleRMSNorm(dim)
        logger.debug(f"SimpleRMSNorm initialized with dim={dim}")

    def forward(
        self, x: TensorType["batch", "seq_len"]
    ) -> TensorType["batch", "seq_len", "vocab_size"]:
        """
        Forward pass of the GSATransformer.

        Args:
            x (TensorType["batch", "seq_len"]): Input tensor of token indices.

        Returns:
            TensorType["batch", "seq_len", "vocab_size"]: Output logits for each token position.
        """
        logger.debug(f"Input shape: {x.shape}")

        x = self.embed(x)
        logger.debug(f"After embedding shape: {x.shape}")

        x = self.norm(x)
        logger.debug(f"After initial norm shape: {x.shape}")

        # Now for each layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            logger.debug(f"After layer {i+1} shape: {x.shape}")

        x = OutputHead(self.dim, vocab_size=self.vocab_size)(x)
        logger.debug(f"Final output shape: {x.shape}")

        return x
