import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cpu"  # mps" if torch.has_mps else "cpu"
eval_iters = 200
embed_dim = 32
# dropout = 0.0

torch.manual_seed(1337)


with open("../local_data/tiny-shakespeare.txt", "r") as f:
    corpus = f.read()

vocab = "".join(sorted(list(set(corpus))))

int_to_str = {i: ch for i, ch in enumerate(vocab)}
str_to_int = {ch: i for i, ch in enumerate(vocab)}


def encode(chars):
    return [str_to_int[ch] for ch in chars]


def decode(ints):
    return "".join([int_to_str[i] for i in ints])


tokens = torch.tensor(encode(corpus), dtype=torch.long)
train = tokens[: int(0.9 * len(tokens))]
valid = tokens[int(0.9 * len(tokens)) :]


def get_batch(split):
    data = train if split == "train" else valid
    rand_idx = torch.randint(len(data) - block_size, (batch_size,))
    x_batch = torch.stack([data[idx : idx + block_size] for idx in rand_idx])
    y_batch = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in rand_idx])
    return x_batch.to(device), y_batch.to(device)


@torch.no_grad()
def estimate_loss(model):
    results = {}
    model.eval()
    for split in ["train", "valid"]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[split] = torch.tensor(losses).mean()
    model.train()
    return results


class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        q, k = self.query(x), self.key(x)  # (B, T, head_size)
        weights = q @ k.transpose(-2, -1) / k.shape[-1] ** 0.5  # (B, T, head_size) @ (B, head_size, T)
        weights = weights.masked_fill(self.tril[: x.shape[1], : x.shape[1]] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        # weights = self.dropout(weights)
        return weights @ self.value(x)  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, embed_dim)

    def forward(self, x):
        return self.projection(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim))

    def forward(self, x):
        return self.layer(x)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        head_size = embed_dim // num_heads
        self.self_attention = MultiHeadAttention(head_size, num_heads)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.self_attention(x)
        x = x + self.feed_forward(x)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_heads=4):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings_table = nn.Embedding(block_size, embed_dim)
        self.attention_blocks = nn.Sequential(
            AttentionBlock(embed_dim, num_heads),
            AttentionBlock(embed_dim, num_heads),
            AttentionBlock(embed_dim, num_heads),
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B, T, embed_dim)
        position_embeddings = self.position_embeddings_table(torch.arange(T, device=device))  # (B, embed_dim)
        x = token_embeddings + position_embeddings  # (B, T, vocab_size)
        x = self.attention_blocks(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (b, c)
            idx_next = torch.multinomial(probs, num_samples=1)  # (b, 1)
            idx = torch.cat([idx, idx_next], dim=1)  # (b, t + 1)
        return idx


model = GPT(len(vocab))
model = model.to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

for step in range(max_iters):

    if step % eval_interval == 0:
        results = estimate_loss(model)
        print(f"step {step}: train loss {results['train']:.4f}, val loss {results['valid']:.4f}")

    x, y = get_batch("train")

    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
