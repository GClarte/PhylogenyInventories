library(ape)

# ── Data ──────────────────────────────────────────────────────────────────────
tree <- read.nexus("Tupi_pruned.nex")
z    <- read.csv("tupi_z.csv")
tree$tip.label

rownames(z) <- z$language
rownames(z)
stopifnot(all(tree$tip.label %in% rownames(z)))

X <- as.matrix(z[tree$tip.label, paste0("z", 0:15)])
X <- scale(X)                             # standardize: diagonal of R ~ 1
N <- nrow(X); p <- ncol(X)

# ── Phylogenetic covariance & whitening ───────────────────────────────────────
C <- vcv(tree)[tree$tip.label, tree$tip.label]
C <- C / mean(diag(C))

L       <- t(chol(C))
Xt      <- solve(L, X)
ot      <- as.vector(solve(L, rep(1, N)))
logdetL <- sum(log(diag(L)))

# ── NIW helpers ───────────────────────────────────────────────────────────────
lmgamma <- function(a, p) {
  p*(p-1)/4 * log(pi) + sum(lgamma(a + (1 - seq_len(p)) / 2))
}

niw_posterior <- function(Xt, ot, mu0, kappa0, Psi0, nu0) {
  N <- nrow(Xt);  p <- ncol(Xt)
  
  s_oX    <- as.vector(crossprod(Xt, ot))   # X'o,  p-vector
  kappa_N <- kappa0 + sum(ot^2)
  nu_N    <- nu0 + N
  mu_N    <- (kappa0 * mu0 + s_oX) / kappa_N
  Psi_N   <- Psi0 + crossprod(Xt) +
    kappa0  * outer(mu0,  mu0) -
    kappa_N * outer(mu_N, mu_N)
  
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
# X is standardized → E[R] = I_p ↔ Psi0 = (nu0-p-1)*I, here = I with nu0 = p+2
mu0    <- rep(0, p)
kappa0 <- 1           # diffuse prior on root state
nu0    <- p + 2       # minimal proper IW (must be > p+1)
Psi0   <- diag(p)     # E[R] = Psi0/(nu0-p-1) = I_p  ✓

# ── Marginal likelihoods ──────────────────────────────────────────────────────
lml_bm  <- log_ml_niw(Xt, ot,      mu0, kappa0, Psi0, nu0) - p * logdetL
lml_iid <- log_ml_niw(X,  rep(1,N), mu0, kappa0, Psi0, nu0)

# ── Bayes Factor ──────────────────────────────────────────────────────────────
log_BF <- lml_bm - lml_iid
cat(sprintf("log BF (BM vs iid) = %7.3f\n", log_BF))
cat(sprintf("    BF (BM vs iid) = %7.3f\n", exp(log_BF)))

# ── Posterior summaries (no MCMC needed) ─────────────────────────────────────
post_bm <- niw_posterior(Xt, ot, mu0, kappa0, Psi0, nu0)

cat("\nPosterior mean of mu (root state):\n");  print(post_bm$mu_N)
cat("\nPosterior mean of R (trait covariance):\n")
print(post_bm$Psi_N / (post_bm$nu_N - p - 1))
