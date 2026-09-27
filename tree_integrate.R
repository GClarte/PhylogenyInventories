library(ape)

# ── Data ──────────────────────────────────────────────────────────────────────
trees <- read.nexus("bantu_posterior_pruned.nex")     # a POSTERIOR SAMPLE (multiPhylo)
z     <- read.csv("atlantic_congo.csv")
rownames(z) <- z$language

# apply the same tip drop to EVERY tree in the sample
if (inherits(trees, "phylo")) trees <- list(trees)          # guard: single tree
trees <- lapply(trees, drop.tip, tip = c("Iñapari"))
class(trees) <- "multiPhylo"
M <- length(trees)
cat(sprintf("integrating over %d posterior trees\n", M))

# fixed tip set (verify all trees share it after dropping)
tips <- trees[[1]]$tip.label
for (m in seq_len(M)) stopifnot(setequal(trees[[m]]$tip.label, tips))
stopifnot(all(tips %in% rownames(z)))

X <- as.matrix(z[tips, paste0("z", 0:15)])
X <- scale(X)                             # standardize once
N <- nrow(X); p <- ncol(X)

# ── NIW helpers (unchanged) ───────────────────────────────────────────────────
lmgamma <- function(a, p) {
  p*(p-1)/4 * log(pi) + sum(lgamma(a + (1 - seq_len(p)) / 2))
}
niw_posterior <- function(Xt, ot, mu0, kappa0, Psi0, nu0) {
  N <- nrow(Xt);  p <- ncol(Xt)
  s_oX    <- as.vector(crossprod(Xt, ot))
  kappa_N <- kappa0 + sum(ot^2)
  nu_N    <- nu0 + N
  mu_N    <- (kappa0 * mu0 + s_oX) / kappa_N
  Psi_N   <- Psi0 + crossprod(Xt) +
    kappa0  * outer(mu0,  mu0) - kappa_N * outer(mu_N, mu_N)
  list(mu_N = mu_N, kappa_N = kappa_N, Psi_N = Psi_N, nu_N = nu_N)
}
log_ml_niw <- function(Xt, ot, mu0, kappa0, Psi0, nu0) {
  N <- nrow(Xt);  p <- ncol(Xt)
  post <- niw_posterior(Xt, ot, mu0, kappa0, Psi0, nu0)
  -N*p/2       * log(pi) +
    p/2         * log(kappa0 / post$kappa_N) +
    nu0/2       * determinant(Psi0,       logarithm = TRUE)$modulus -
    post$nu_N/2 * determinant(post$Psi_N, logarithm = TRUE)$modulus +
    lmgamma(post$nu_N/2, p) - lmgamma(nu0/2, p)
}

# ── Prior hyperparameters ─────────────────────────────────────────────────────
mu0    <- rep(0, p)
kappa0 <- 1
nu0    <- p + 2
Psi0   <- diag(p)

# ── iid marginal likelihood: computed ONCE (tree-independent) ─────────────────
lml_iid <- log_ml_niw(X, rep(1, N), mu0, kappa0, Psi0, nu0)

# ── BM marginal likelihood for EVERY tree ─────────────────────────────────────
lml_bm <- numeric(M)
for (m in seq_len(M)) {
  tr <- trees[[m]]
  C  <- vcv(tr)[tips, tips]                 # REORDER to match X rows
  C  <- C / mean(diag(C))
  L       <- t(chol(C))
  Xt      <- solve(L, X)
  ot      <- as.vector(solve(L, rep(1, N)))
  logdetL <- sum(log(diag(L)))
  lml_bm[m] <- log_ml_niw(Xt, ot, mu0, kappa0, Psi0, nu0) - p * logdetL
}

# ── Integrate over trees: natural-scale mean via logsumexp ────────────────────
logsumexp <- function(x){ mx <- max(x); mx + log(sum(exp(x - mx))) }
lml_bm_int <- logsumexp(lml_bm) - log(M)

log_BF <- lml_bm_int - lml_iid
cat(sprintf("\nINTEGRATED log BF (BM vs iid) = %7.3f\n", log_BF))
cat(sprintf("        integrated BF          = %.3e\n", exp(log_BF)))

# ── How much does tree uncertainty matter? ────────────────────────────────────
per_tree_logBF <- lml_bm - lml_iid
cat("\nper-tree log BF distribution:\n"); print(summary(per_tree_logBF))
cat(sprintf("per-tree logml_bm range (max-min): %.2f\n", max(lml_bm)-min(lml_bm)))
cat(sprintf("single-best-tree log BF:           %.3f\n", max(per_tree_logBF)))