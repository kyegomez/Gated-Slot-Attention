

# Gated Slot Attention

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)



## Install
```bash
pip3 install -U gated-slot-attention
```



## Usage
For full usage, use your own tokenizer and vocab size.

```python
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

```

# License
MIT
