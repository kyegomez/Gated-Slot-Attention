import torch
from gated_slot_attention.model import GSATransformer

model = GSATransformer(
    dim=512,
    heads=8,
    m=64,
    tau=0.1,
    depth=1,
    vocab_size=10000,
    max_seq_len=1024,
)

x = torch.randint(0, 10000, (1, 1024))
out = model(x)
print(out.shape)
