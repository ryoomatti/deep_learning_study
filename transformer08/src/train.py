
import math
import random
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")



##Self-Attention##

class SelfAttention(nn.Module):

    def __init__(self, config, resid_pdrop=0.1, attn_pdrop=0.1, causal=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.causal = causal
        if causal:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
            )

        self.n_head = config.n_head

    def forward(self, x):
        b, t, d = x.size()

        k = self.key(x).view(b, t, self.n_head, d // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, d // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, d // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.causal:
            att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)  # key方向(最後の軸)に正規化　各クエリがどのkeyに注目しているか＃
        att = self.attn_drop(att)

        y = att @ v #全部のトークンのValueが、その注目度の割合ぶんだけブレンドされて出てくる#
        y = y.transpose(1, 2).contiguous().view(b, t, d)

        y = self.resid_drop(self.proj(y))
        return y



##Transformer Block (Attention + MLP)##

class Block(nn.Module):
    def __init__(self, config, resid_pdrop=0.1, causal=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config, causal=causal)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = self.attn(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


##GPT本体##
class GPT(nn.Module):
    def __init__(self, config, embd_pdrop=0.1):
        super().__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params/1e6:.2f}M")

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #各トークンに対する次の確率がtargetに対してどれだけ割わてられているかを-log()でlossとして計算#

        return logits, loss

    def configure_optimizers(self, learning_rate=3e-4, weight_decay=0.1, betas=(0.9, 0.95)):
      
        decay, no_decay = set(), set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
        no_decay.add("pos_emb")

        param_dict = dict(self.named_parameters())
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


##文字レベルデータセット##
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"Data has {data_size} characters, {vocab_size} unique.")

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) #一個ずらしでIDを撮ってくる#
        return x, y

##学習ループ##

def train(
    model,
    train_dataset,
    test_dataset=None,
    max_epochs=10,
    batch_size=64,
    learning_rate=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    num_workers=0,
    ckpt_path=None,
):
    

    model = model.to(device)
    optimizer = model.configure_optimizers(learning_rate, weight_decay, betas)

    def run_epoch(dataset, is_train, epoch):
        model.train(is_train)
        loader = DataLoader(
            dataset, shuffle=is_train, batch_size=batch_size, num_workers=num_workers
        )

        losses = []
        pbar = tqdm(loader, desc=f"epoch {epoch+1} ({'train' if is_train else 'test'})")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            with torch.set_grad_enabled(is_train):
                logits, loss = model(x, y)
                losses.append(loss.item())

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return float(np.mean(losses))

    best_loss = float("inf")
    for epoch in range(max_epochs):
        train_loss = run_epoch(train_dataset, is_train=True, epoch=epoch)
        logger.info("epoch %d: train loss %f", epoch + 1, train_loss)

        if test_dataset is not None:
            test_loss = run_epoch(test_dataset, is_train=False, epoch=epoch)
            logger.info("epoch %d: test loss %f", epoch + 1, test_loss)
            good_model = test_loss < best_loss
            if ckpt_path is not None and good_model:
                best_loss = test_loss
                torch.save(model.state_dict(), ckpt_path)
                logger.info("saving checkpoint to %s", ckpt_path)
        elif ckpt_path is not None:
            torch.save(model.state_dict(), ckpt_path)

    return model


##文章生成##

def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, do_sample=False, top_k=None):

    block_size = model.get_block_size()
    model.eval()
    x = x.to(device)

    for _ in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)

        if do_sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        x = torch.cat((x, ix), dim=1)

    return x


##デモ##

if __name__ == "__main__":
    class GPTConfig:
        def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
            self.vocab_size = vocab_size
            self.block_size = block_size
            self.n_layer = n_layer
            self.n_head = n_head
            self.n_embd = n_embd

    text = "hello world, this is a tiny example corpus for a character-level GPT model. " * 50
    block_size = 32

    dataset = CharDataset(text, block_size)
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=64,
    )
    model = GPT(config)

    train(model, dataset, max_epochs=5, batch_size=16)

    context = "hello"
    x = torch.tensor([dataset.stoi[c] for c in context], dtype=torch.long)[None, ...]
    y = sample(model, x, steps=100, temperature=1.0, do_sample=True, top_k=10)
    completion = "".join([dataset.itos[int(i)] for i in y[0]])
    print(completion)