library(phytools)
library(ape)

tree <- read.nexus("PN_pruned_2.nex")    
z <- read.csv("pn_z.csv")
rownames(z) <- z$language                      

# align z to tree tips
stopifnot(all(tree$tip.label %in% rownames(z)))
X <- as.matrix(z[tree$tip.label, paste0("z", 0:15)])
X <- scale(X)                                   

# --- Pagel's lambda, per dimension, with significance test ---
cat("=== Pagel's lambda ===\n")
lambda_res <- lapply(0:15, function(i) {
  trait <- setNames(X[, i+1], rownames(X))
  ps <- phylosig(tree, trait, method = "lambda", test = TRUE)
  cat(sprintf("  z%-2d: lambda = %.3f   p = %.4f\n", i, ps$lambda, ps$P))
  ps
})

# --- Blomberg's K, per dimension, with significance test ---
cat("\n=== Blomberg's K ===\n")
K_res <- lapply(0:15, function(i) {
  trait <- setNames(X[, i+1], rownames(X))
  ps <- phylosig(tree, trait, method = "K", test = TRUE, nsim = 1000)
  cat(sprintf("  z%-2d: K = %.3f   p = %.4f\n", i, ps$K, ps$P))
  ps
})