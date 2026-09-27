import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from scipy.spatial.distance import pdist, squareform


merged = pd.read_csv("z_with_families.csv")

zcols = sorted([c for c in merged.columns if c.startswith("z") and c[1:].isdigit()],
               key=lambda c: int(c[1:]))
print(f"{len(merged)} languages, {len(zcols)} z-dims, "
      f"{merged['Family_ID'].nunique()} families")

assert np.isfinite(merged[zcols].values).all(), "non-finite z in CSV!"

fam_sizes = (merged.groupby("Family_ID")["size"]
                   .agg(n="count", mean="mean", std="std", min="min", max="max")
                   .sort_values("n", ascending=False))
print("\n--- size distribution by family (top 20) ---")
print(fam_sizes.head(20).to_string())

big_fams = fam_sizes[fam_sizes["n"] >= 5].index
groups = [merged.loc[merged["Family_ID"] == f, "size"].values for f in big_fams]

print(f"\n{len(big_fams)} families with >=5 members (used for the test)")
if len(groups) >= 2:
    F, p_anova = f_oneway(*groups)
    H, p_kw    = kruskal(*groups)
    print(f"size differs by family?  ANOVA F={F:.2f} p={p_anova:.2e}   "
          f"Kruskal H={H:.2f} p={p_kw:.2e}")
else:
    p_anova = np.nan
    print("Not enough families with >=5 members for a size-by-family test.")


size = merged["size"].values.astype(float)
Z    = merged[zcols].values

DEG = 3                                          # cubic fit of z on size

def poly_r2(x, y, deg=DEG):
    """R² of a degree-`deg` polynomial fit of y on x."""
    coeffs = np.polyfit(x, y, deg)               # fit
    y_hat  = np.polyval(coeffs, x)               # predict
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return (1 - ss_res / ss_tot if ss_tot > 0 else 0.0), y_hat

print(f"\n--- variance of each z-dim explained by inventory size (R², deg={DEG}) ---")
r2_per_dim = []
pred = np.zeros_like(Z)
for j, c in enumerate(zcols):
    r2, y_hat = poly_r2(size, Z[:, j])
    r2_per_dim.append(r2)
    pred[:, j] = y_hat
    print(f"  {c}: R² = {r2:.3f}")

# overall share of TOTAL z variance explained by size
ss_res   = np.sum((Z - pred) ** 2)
ss_tot   = np.sum((Z - Z.mean(0)) ** 2)
total_r2 = 1 - ss_res / ss_tot
print(f"\nmax per-dim R² = {max(r2_per_dim):.3f} "
      f"(dim {zcols[int(np.argmax(r2_per_dim))]})")
print(f"TOTAL z variance explained by size: {total_r2:.3f}")

pred = np.column_stack([
    np.polyval(np.polyfit(size, Z[:, j], DEG), size) for j in range(Z.shape[1])
])
Z_resid = Z - pred

# --- filter to families with enough members
MIN_MEMBERS = 5
fam_counts = merged["Family_ID"].value_counts()
keep_fams = fam_counts[fam_counts >= MIN_MEMBERS].index
mask = merged["Family_ID"].isin(keep_fams).values
labels = merged.loc[mask, "Family_ID"].values

print(f"{mask.sum()} languages across {len(keep_fams)} families "
      f"(>= {MIN_MEMBERS} members each)")

def standardize(A):
    return (A - A.mean(0)) / (A.std(0) + 1e-9)

def silhouette_by_label(X, labels):
    """Mean silhouette using family as the cluster label."""
    D = squareform(pdist(X))                      # pairwise distances
    uniq = np.unique(labels)
    sil = np.zeros(len(labels))
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False                           # exclude self
        # a(i): mean dist to own family
        a = D[i, same].mean() if same.sum() > 0 else 0.0
        # b(i): min over other families of mean dist to that family
        b = np.inf
        for f in uniq:
            if f == labels[i]:
                continue
            other = labels == f
            if other.sum() > 0:
                b = min(b, D[i, other].mean())
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return sil.mean(), sil

def report_silhouette(name, X, labels):
    Xs = standardize(X)
    mean_sil, _ = silhouette_by_label(Xs, labels)
    print(f"  {name:<12} mean silhouette = {mean_sil:+.4f}")
    return mean_sil

print("\n--- Silhouette (family as cluster; higher = tighter family grouping) ---")
print("  (range -1..+1; ~0 = no family structure; >0 = families group)")
sil_raw   = report_silhouette("raw z",   Z[mask],       labels)
sil_resid = report_silhouette("resid z", Z_resid[mask], labels)

def distance_gap_test(X, labels, n_perm=2000, seed=0):
    """Statistic = mean(cross-family dist) - mean(same-family dist).
       Positive => same-family pairs are closer. Permutation p-value."""
    rng = np.random.default_rng(seed)
    Xs = standardize(X)
    D = squareform(pdist(Xs))
    iu = np.triu_indices(len(labels), k=1)          # unique pairs
    dists = D[iu]
    same_pair = (labels[iu[0]] == labels[iu[1]])

    def stat(sp):
        return dists[~sp].mean() - dists[sp].mean()

    obs = stat(same_pair)
    perm = np.empty(n_perm)
    lab = labels.copy()
    for k in range(n_perm):
        rng.shuffle(lab)
        sp = (lab[iu[0]] == lab[iu[1]])
        perm[k] = stat(sp)
    p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    return obs, p, same_pair, dists

def report_gap(name, X, labels):
    obs, p, same_pair, dists = distance_gap_test(X, labels)
    print(f"  {name:<12} gap = {obs:+.4f}  (same μ={dists[same_pair].mean():.3f}, "
          f"cross μ={dists[~same_pair].mean():.3f})  perm p = {p:.4f}")
    return obs, p

print("\n--- Same-family vs cross-family distance (permutation, 2000 reps) ---")
gap_raw   = report_gap("raw z",   Z[mask],       labels)
gap_resid = report_gap("resid z", Z_resid[mask], labels)