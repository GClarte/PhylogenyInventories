import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy.optimize import linear_sum_assignment
import json

BOTTLENECK_SIZE = 16
torch.manual_seed(1337)

phoible = pd.read_csv("full_phoible.csv", index_col = 0)
phoible = phoible[phoible["SegmentClass"] != "tone"]
#phoible = phoible[phoible["SegmentClass"] == "vowel"]
phoible = phoible[phoible["Marginal"] != "TRUE"]

PHOIBLE_FEATURES = ["stress","syllabic","short","long","consonantal","sonorant",
    "continuant","delayedRelease","approximant","tap","trill","nasal","lateral",
    "labial","round","labiodental","coronal","anterior","distributed","strident",
    "dorsal","high","low","front","back","tense","retractedTongueRoot",
    "advancedTongueRoot","periodicGlottalSource","epilaryngealSource",
    "spreadGlottis","constrictedGlottis","fortis","raisedLarynxEjective",
    "loweredLarynxImplosive","click",
]

feat_cols = [c for c in PHOIBLE_FEATURES if c in phoible.columns]
print(f"{len(feat_cols)} feature columns")

def encode_cell(val, symbols=('+', '-', '0')):
    parts = set(p.strip() for p in str(val).split(','))
    return [1.0 if s in parts else 0.0 for s in symbols]

def encode_phoneme(row):
    out = []
    for c in feat_cols:
        out.extend(encode_cell(row[c]))
    return out                                  # length = 3 * len(feat_cols)

IN_DIM = 3 * len(feat_cols)                      # ~111  <-- new in_dim
print("in_dim:", IN_DIM)

GROUP_KEY = "InventoryID"          # one inventory per doculect

sequences, lengths, inv_ids = [], [], []
for inv_id, grp in phoible.groupby(GROUP_KEY):
    mat = np.array([encode_phoneme(r) for _, r in grp.iterrows()], dtype=float)
    mat = np.unique(mat, axis=0)                # dedupe identical segments within inventory
    if mat.shape[0] == 0:
        continue
    sequences.append(torch.tensor(mat, dtype=torch.float32))
    lengths.append(torch.tensor(mat.shape[0], dtype=torch.int))
    inv_ids.append(inv_id)

print(f"{len(sequences)} inventories")
MAX_LENGTH  = max(int(l) for l in lengths)
NUM_QUERIES = MAX_LENGTH + 4
print("max inventory size:", MAX_LENGTH)

all_segs = np.concatenate([s.numpy() for s in sequences], axis=0)
uniq = np.unique(all_segs, axis=0)
V = uniq.shape[0]

PAD_ID, START_ID = 0, 1
seg_to_id = {tuple(row): i + 2 for i, row in enumerate(uniq)}
id_to_seg = torch.tensor(uniq, dtype=torch.float32)
vocab_total = V + 2
EMPTY_IDX   = vocab_total
print(f"{V} distinct segments, vocab_total={vocab_total}")

def to_ids(arr):
    return torch.tensor([seg_to_id[tuple(r)] for r in arr], dtype=torch.long)
ids = [to_ids(s.numpy()) for s in sequences]

class SequenceDataset(Dataset):
    def __init__(self, sequences, ids, lengths):
        self.sequences = sequences
        self.lengths = lengths
        self.ids = ids
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.ids[idx], self.lengths[idx]

def collate_fn(batch):
    seqs, id_seqs, lengths = zip(*batch)
    feats   = pad_sequence(seqs, batch_first=True)                 # (B, n, 23)
    tgt_ids = pad_sequence(id_seqs, batch_first=True,
                           padding_value=PAD_ID)                   # (B, n)
    return feats, tgt_ids, torch.stack(lengths)

dataset = SequenceDataset(sequences, ids, lengths)
train_size = int(0.8 * len(dataset)); val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=collate_fn)



class SetEncoder(nn.Module):
    def __init__(self, in_dim=IN_DIM, d_model=256, nhead=4, layers=3, d_z=8):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                         dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.to_z = nn.Linear(d_model, d_z)

    def forward(self, x, lengths):
        lengths = lengths.to(x.device)
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        h = self.encoder(self.embed(x), src_key_padding_mask=mask)
        m = (~mask).unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1) # mean pooling
        pooled = self.norm(pooled)
        return self.to_z(pooled)

class QuerySetDecoder(nn.Module):
    def __init__(self, vocab_size, d_z=8, d_model=256, nhead=4,
                 layers=3, num_queries=NUM_QUERIES):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.z_proj  = nn.Linear(d_z, d_model)
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=256,
                                         dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.out = nn.Linear(d_model, vocab_size + 1)

    def forward(self, z):
        mem = self.z_proj(z).unsqueeze(1)
        q = self.queries.unsqueeze(0).expand(z.size(0), -1, -1)
        h = self.decoder(q, mem)
        return self.out(h)                                    # (B, Q, V+1)

class SetAutoencoder(nn.Module):
    def __init__(self, vocab_size, in_dim=IN_DIM, d_z=8, num_queries=NUM_QUERIES):
        super().__init__()
        self.encoder = SetEncoder(in_dim, d_z=d_z)
        self.decoder = QuerySetDecoder(vocab_size=vocab_size, d_z=d_z,
                                       num_queries=num_queries)          # <-- vocab_size, not out_dim
    def forward(self, x, lengths):
        z = self.encoder(x, lengths)
        logits = self.decoder(z)          # (B, Q, vocab_size+1)  single tensor
        return logits, z                  # <-- no pres_logits

def hungarian_token_loss(logits, tgt_ids, lengths, empty_idx, empty_weight=0.5):
    B, Q, Vp1 = logits.shape
    device = logits.device
    weight = torch.ones(Vp1, device=device); weight[empty_idx] = empty_weight
    logp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for b in range(B):
        n = int(lengths[b])
        full_tgt = torch.full((Q,), empty_idx, device=device)
        if n > 0:
            ids = tgt_ids[b, :n]
            cost = -logp[b][:, ids]                           # (Q, n)
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            row = torch.as_tensor(row, device=device)
            col = torch.as_tensor(col, device=device)
            full_tgt[row] = ids[col]
        total = total + F.cross_entropy(logits[b], full_tgt, weight=weight)
    return total / B

@torch.no_grad()
def segment_scores(model, feats, lengths):
    logits, z = model(feats, lengths)
    probs = logits.softmax(-1)[..., :-1]     # drop ∅ -> (B, vocab_total)
    return probs.max(dim=1).values           # (B, vocab_total); real segs at idx >=2

@torch.no_grad()
def predict_sets(model, feats, lengths):
    logits, z = model(feats, lengths)
    pred = logits.softmax(-1).argmax(-1)                     # (B, Q)
    keep = (pred != EMPTY_IDX) & (pred != PAD_ID) & (pred != START_ID)
    return [set(pred[b][keep[b]].tolist()) for b in range(pred.size(0))], z

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SetAutoencoder(vocab_size=vocab_total, d_z=BOTTLENECK_SIZE, num_queries=NUM_QUERIES).to(device) # d_z is bottleneck size
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)




@torch.no_grad()
def target_sets(tgt_ids, lengths):
    return [set(tgt_ids[b, :int(lengths[b])].tolist()) for b in range(tgt_ids.size(0))]

@torch.no_grad()
def set_f1(pred, tgt):
    tp=fp=fn=0
    for p,t in zip(pred,tgt):
        tp+=len(p&t); fp+=len(p-t); fn+=len(t-p)
    pr = tp/(tp+fp+1e-9); rc = tp/(tp+fn+1e-9)
    f1 = 2*pr*rc/(pr+rc+1e-9)
    print(f"precision {pr:.3f}  recall {rc:.3f}  f1 {f1:.3f}  TP {tp} FP {fp} FN {fn}")
    return f1

@torch.no_grad()
def evaluate(model, loader, shuffle_z=False):
    model.eval(); preds=[]; tgts=[]
    for feats, tgt_ids, lengths in loader:
        feats, lengths = feats.to(device), lengths.to(device)
        z = model.encoder(feats, lengths)
        if shuffle_z: z = z[torch.randperm(z.size(0))]
        pred = model.decoder(z).softmax(-1).argmax(-1)
        keep = (pred!=EMPTY_IDX)&(pred!=PAD_ID)&(pred!=START_ID)
        preds += [set(pred[b][keep[b]].tolist()) for b in range(pred.size(0))]
        tgts  += target_sets(tgt_ids, lengths)
    return set_f1(preds, tgts)


EPOCHS = 200
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4,      # was 3e-4
            steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.25)  # was 0.1

best_f1, patience, wait = -1.0, 20, 0
for epoch in range(EPOCHS):
    model.train()
    for feats, tgt_ids, lengths in train_loader:
        feats, tgt_ids, lengths = feats.to(device), tgt_ids.to(device), lengths.to(device)
        logits, z = model(feats, lengths)
        loss = hungarian_token_loss(logits, tgt_ids, lengths, EMPTY_IDX)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    val_f1 = evaluate(model, val_loader)
    if val_f1 > best_f1:
        best_f1 = val_f1; wait = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print(f"early stop at epoch {epoch} (no val improvement for {patience} epochs)")
            break
    print(f"epoch {epoch}  val_F1 {val_f1:.4f}  best {best_f1:.4f}  wait {wait}")

model.load_state_dict(torch.load("best_model.pt", weights_only=True))

torch.save(model.state_dict(), "full_model_randinv.pt")

print("train F1   :", evaluate(model, train_loader))
print("val F1     :", evaluate(model, val_loader))
print("val F1 (z⊥):", evaluate(model, val_loader, shuffle_z=True))

