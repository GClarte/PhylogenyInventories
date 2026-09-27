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



model.load_state_dict(torch.load("full_model (z=16).pt", weights_only=True))

print("train F1   :", evaluate(model, train_loader))
print("val F1     :", evaluate(model, val_loader))
print("val F1 (z⊥):", evaluate(model, val_loader, shuffle_z=True))


ST = {
    "amdo1237": "Amdo Tibetan",              # ✓
    "belh1239": "Belhariya",                 # ✓
    "nucl1310": "Burmese",                   # ✓
    "chep1245": "Chepang",                   # ✓
    "tang1336": "East-Central Tangkhul Naga",# ✓ verify
    "garo1247": "Garo",                      # ✓
    "haka1240": "Haka Chin",                 # ✓ (Hakha Chin)
    "hakk1236": "Hakka Chinese",             # ✓
    "karb1241": "Hills Karbi",               # ✓ verify (Karbi/Mikir)
    "kham1282": "Khams Tibetan",            # ✓  verify
    "kulu1253": "Kulung (Nepal)",            # ✓
    "lisu1250": "Lisu",                      # ✓
    "acha1249": "Longchuan Achang",          # ✓ verify (Achang; Longchuan dialect)
    "mand1415": "Mandarin Chinese",          # ✓
    "minn1241": "Min Nan Chinese",           # ✓
    "lush1249": "Mizo",                      # ✓ (Lushai/Mizo)
    "kach1280": "Southern Jinghpaw",         # ✓ verify (Jingpho)
    "thul1246": "Thulung",                   # ✓
    "tibe1272": "Tibetan",                   # ✓ broad code — may not match a specific PHOIBLE row
    "tsha1245": "Tshangla",                  # ✓
    "wayu1241": "Wayu",                      # ✓ verify (Wayu/Hayu)
    "yuec1235": "Yue Chinese",               # ✓
    "zaiw1241": "Zaiwa",                     # ✓ (Zaiwa/Atsi)
}

Yupi = {
    "ache1246": "Aché",
    "araw1273": "Araweté",
    "avac1239": "Avá-Canoeiro",
    "awet1244": "Awetí",
    "nhan1238": "Chiripá",
    "coca1259": "Cocama-Cocamilla",
    "guaj1256": "Guajá",
    "guar1292": "Guarayu",
    "kaiw1246": "Kaiwá",
    "kama1373": "Kamayurá",
    "kaya1329": "Kayabí",
    "mbya1239": "Mbyá Guaraní",
    "omag1248": "Omagua",
    "para1311": "Paraguayan Guaraní",
    "para1312": "Parakanã",
    "sate1243": "Sateré-Mawé",
    "siri1273": "Sirionó",
    "tapi1253": "Tapieté",
    "tapi1254": "Tapirapé",
    "emer1243": "Teko",
    "tenh1241": "Tenharim-Parintintin-Diahoi",
    "toca1235": "Tocantins Asurini",
    "tupi1273": "Tupinambá",
    "urub1250": "Urubú-Kaapor",
    "waya1269": "Wayampi",
    "xeta1241": "Xetá",
    "xing1248": "Xingú Asuriní",
    "yuqu1240": "Yuqui",
}

Araw = {
    "acha1250": "Achagua",
    "apur1254": "Apurinã",
    "asha1243": "Asháninka",
    "guar1293": "Baniva de Maroa",
    "bani1255": "Baniwa do Icana",
    "bare1276": "Baré",
    "baur1253": "Baure",
    "cabi1241": "Cabiyarí",
    "caqu1242": "Caquinte",
    "cham1318": "Chamicuro",
    "curr1243": "Curripaco",
    "enaw1238": "Enawené-Nawé",
    "igna1246": "Ignaciano",
    "inap1242": "Iñapari",
    "isla1278": "Island Carib",
    "guan1270": "Kinikinao",
    "loko1255": "Lokono",
    "mach1267": "Machiguenga",
    "mach1268": "Machinere",
    "mehi1240": "Mehináku",
    "noma1263": "Nomatsiguenga",
    "pali1279": "Palikúr",
    "para1316": "Paraujano",
    "pare1272": "Parecís",
    "piap1246": "Piapoco",
    "resi1247": "Resígaro",
    "tari1256": "Tariana",
    "tere1279": "Terena-Kinikinao-Chane",
    "trin1274": "Trinitario-Javeriano-Loretano",
    "wapi1253": "Wapishana",
    "waur1244": "Waurá",
    "wayu1243": "Wayuu",
    "yane1238": "Yanesha",
    "yavi1244": "Yavitero-Pareni",
    "yawa1261": "Yawalapití",
    "yine1238": "Yine",
    "yucu1253": "Yucuna",
}


target_glottocodes =  {'aghe1239': 'Aghem', 'bafi1243': 'Bafia',
                       'beem1239': 'Beembe',
                       'bila1255': 'Bila', 'buku1249': 'Bukusu', 'digo1243': 'Digo', 'dual1243': 'Duala',
                       'ewon1239': 'Ewondo', 'fipa1238': 'Fipa', 'giry1241': 'Giryama', 'haya1250': 'Haya', 'here1253':
                        'Herero', 'holo1240': 'Holoholo', 'jita1239': 'Jita', 'kagu1239': 'Kagulu',
                       'kako1242': 'Kako', 'kiku1240': 'Kikuyu', 'kuri1259': 'Kuria', 'lang1320': 'Langi',
                       'lega1249': 'Lega-Shabunda', 'leng1258': 'Lengola', 'lund1266': 'Lunda', 'luva1239': 'Luvale',
                       'mach1266': 'Machame', 'mokp1239': 'Mokpwe', 'mung1266': 'Mungaka', 'ndam1239': 'Ndamba',
                       'ndon1254': 'Ndonga', 'okuu1243': 'Oku', 'pang1287': 'Pangwa', 'rund1242': 'Rundi', 'shon1251': 'Shona', 'suku1261': 'Sukuma', 'swah1253': 'Swahili', 'tivv1240': 'Tiv', 'tuki1240': 'Tuki', 'umbu1257': 'Umbundu', 'wong1247': 'Wongo', 'xhos1239': 'Xhosa',
                       'yaoo1241': 'Yao', 'zulu1248': 'Zulu', 'basa1284': 'Basa (Cameroon)', 'bemb1257': 'Bemba (Zambia)', 'bena1262': 'Bena (Tanzania)', 'bulu1251': 'Bulu (Cameroon)', 'fang1248': 'Fang (Cameroon)', 'fefe1239': "Fefe",
                       'gand1255': 'Ganda', 'gwen1239': 'Gweno', 'hang1260': 'Hangaza', 'kahe1238': 'Kahe', 'ikal1242': 'Kalanga', 'kamb1297': 'Kamba (Kenya)', 'komc1235': 'Kom (Cameroon)', 'mako1251': 'Makonde', 'meru1245': 'Meru', 'moch1256': 'Mochi',
                       'mong1338': 'Mongo', 'mwan1247': 'Mwani', 'ndal1241': 'Ndali', 'ndzw1235': 'Ndzwani Comorian', 'nugu1242': 'Nugunu (Cameroon)', 'nyam1277': 'Nyambo', 'nyam1276': 'Nyamwezi', 'sumb1240': 'Sumbwa',
     'zinz1238': 'Zinza'}

matched = phoible[phoible["Glottocode"].isin(target_glottocodes.keys())]

found = set(matched["Glottocode"])
missing = {gc: name for gc, name in target_glottocodes.items() if gc not in found}

print(f"matched {len(found)}/{len(target_glottocodes)} glottocodes")
if missing:
    print("\nNOT found in PHOIBLE (verify these glottocodes):")
    for gc, name in missing.items():
        print(f"    {gc}  {name}")


@torch.no_grad()
def encode_matrix(mat):
    feats  = torch.tensor(mat, dtype=torch.float32).unsqueeze(0).to(device)
    length = torch.tensor([mat.shape[0]], dtype=torch.int).to(device)
    return model.encoder(feats, length).squeeze(0).cpu().numpy()

records = []
for gc, grp_lang in matched.groupby("Glottocode"):
    label  = target_glottocodes[gc]
    chosen = grp_lang.groupby("InventoryID").size().idxmax()      # largest inventory
    grp    = grp_lang[grp_lang["InventoryID"] == chosen]

    mat = np.array([encode_phoneme(r) for _, r in grp.iterrows()], dtype=float)
    mat = np.unique(mat, axis=0)
    if mat.shape[0] == 0:
        continue
    z = encode_matrix(mat)
    records.append({
        "language": label, "glottocode": gc,
        "iso": grp["ISO6393"].iloc[0] if "ISO6393" in grp else "",
        "inventory_id": chosen, "size": mat.shape[0],
        **{f"z{i}": z[i] for i in range(z.shape[0])},
    })

zdf = pd.DataFrame(records)
assert np.isfinite(zdf.filter(regex=r'^z\d+$').values).all(), "non-finite z!"
zdf.to_csv("atlantic_congo.csv", index=False, encoding="utf-8-sig")
print(f"\nencoded {len(zdf)} languages -> atlantic_congo.csv "
      f"(z dims = {zdf.filter(regex=r'^z[0-9]+$').shape[1]})")

