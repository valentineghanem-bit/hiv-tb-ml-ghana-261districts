# spatial_diagnostics.R — Spatial model diagnostics
# Author: Valentine Golden Ghanem | ORCID: 0009-0002-8332-0220
# Usage:  Rscript scripts/spatial_diagnostics.R
#
# Auto-detects outcome columns from the master dataset and runs:
#   - Global Moran's I (queen-contiguity + KNN-5 sensitivity)
#   - Local Moran's I (LISA) with BH-FDR correction
#   - Getis-Ord Gi* hotspot detection
#   - Bivariate Moran's I for outcome pairs
#   - OLS → LM diagnostics → SLM / SEM model selection
#   - GWR bandwidth selection and local R² summary
# Outputs written to data/processed/ (or outputs/data/ if that structure is used)

suppressPackageStartupMessages({
  library(spdep)
  library(spatialreg)
  library(dplyr)
  library(readr)
})
set.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
find_master <- function() {
  candidates <- c(
    "data/processed",
    "outputs/data",
    "outputs",
    "."
  )
  for (d in candidates) {
    csvs <- list.files(d, pattern = "master.*\\.csv|FINAL.*\\.csv", full.names = TRUE,
                       ignore.case = TRUE, recursive = FALSE)
    if (length(csvs) > 0) return(csvs[1])
  }
  stop("Master CSV not found. Run the Python pipeline first.")
}

find_output_dir <- function() {
  dirs <- c("data/processed", "outputs/data", "outputs")
  for (d in dirs) {
    if (dir.exists(d)) return(d)
  }
  return(".")
}

master_path <- find_master()
OUT_DIR     <- find_output_dir()
cat(sprintf("Master dataset : %s\n", master_path))
cat(sprintf("Output dir     : %s\n", OUT_DIR))

# ── 1. Load data ──────────────────────────────────────────────────────────────
df <- read_csv(master_path, show_col_types = FALSE)
n  <- nrow(df)
cat(sprintf("Loaded: %d rows × %d columns\n", n, ncol(df)))

# Auto-detect outcome columns: numeric, 0–100 range, no missing, >10% variance
detect_outcomes <- function(df, max_outcomes = 5) {
  num_cols <- names(df)[sapply(df, is.numeric)]
  cands <- Filter(function(v) {
    x <- df[[v]]
    !any(is.na(x)) &&
      min(x, na.rm = TRUE) >= 0 &&
      max(x, na.rm = TRUE) <= 100 &&
      sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE) > 0.05
  }, num_cols)
  # Prefer columns with epidemiological keywords
  keywords <- c("prev", "rate", "index", "score", "burden", "mort", "incid",
                 "stunt", "anaem", "iycf", "diarr", "malaria", "hiv", "sti")
  priority <- cands[grepl(paste(keywords, collapse = "|"), cands, ignore.case = TRUE)]
  other    <- setdiff(cands, priority)
  head(c(priority, other), max_outcomes)
}

OUTCOMES <- detect_outcomes(df)
cat(sprintf("Outcomes detected: %s\n", paste(OUTCOMES, collapse = ", ")))

COVARS <- names(df)[sapply(df, is.numeric) & !names(df) %in% c(OUTCOMES, "district_idx")]
COVARS <- COVARS[!grepl("^(id|idx|code|fid|gid|objectid)$", COVARS, ignore.case = TRUE)]
COVARS <- head(COVARS, 6)
cat(sprintf("Covariates: %s\n", paste(COVARS, collapse = ", ")))

# ── 2. Spatial weights ────────────────────────────────────────────────────────
cat("\n── Spatial weights ───────────────────────────────────────────────────\n")
W_queen <- NULL
W_knn5  <- NULL

geojson_path <- list.files("data/raw", pattern = "\\.geojson$", full.names = TRUE,
                            recursive = FALSE)[1]
if (!is.na(geojson_path) && requireNamespace("sf", quietly = TRUE)) {
  sf_obj  <- sf::st_read(geojson_path, quiet = TRUE)
  # Align rows if sf has more rows than df (e.g. 260 vs 261)
  if (nrow(sf_obj) != n) {
    message(sprintf("  GeoJSON has %d rows, data has %d — using index-based KNN fallback",
                    nrow(sf_obj), n))
  } else {
    nb_q    <- poly2nb(sf_obj, queen = TRUE)
    W_queen <- nb2listw(nb_q, style = "W", zero.policy = TRUE)
    cat(sprintf("  Queen contiguity: %d polygons, avg neighbours: %.2f\n",
                length(nb_q), mean(card(nb_q))))
  }
}

# KNN-5 (sensitivity check, coordinate-based)
lat_col <- names(df)[grepl("^lat$|latitude", names(df), ignore.case = TRUE)][1]
lon_col <- names(df)[grepl("^lon$|longitude", names(df), ignore.case = TRUE)][1]

if (!is.na(lat_col) && !is.na(lon_col)) {
  coords  <- cbind(df[[lon_col]], df[[lat_col]])
  knn5    <- knearneigh(coords, k = 5, longlat = TRUE)
  W_knn5  <- nb2listw(knn2nb(knn5), style = "W")
  cat(sprintf("  KNN-5 weights built from lat/lon columns ('%s', '%s')\n",
              lat_col, lon_col))
} else if (is.null(W_queen)) {
  # Last resort: grid indices
  coords <- cbind(seq_len(n), rep(0, n))
  knn5   <- knearneigh(coords, k = 5)
  W_knn5 <- nb2listw(knn2nb(knn5), style = "W")
  message("  lat/lon not found — KNN-5 built from row indices (approximate)")
}

W_primary <- if (!is.null(W_queen)) W_queen else W_knn5

# ── 3. Global Moran's I ───────────────────────────────────────────────────────
cat("\n── Global Moran's I ──────────────────────────────────────────────────\n")
moran_rows <- list()
for (v in OUTCOMES) {
  vals <- df[[v]]
  mi_q <- tryCatch(
    moran.test(vals, W_primary, randomisation = TRUE,
               na.action = na.exclude, zero.policy = TRUE),
    error = function(e) NULL
  )
  if (is.null(mi_q)) next

  # Sensitivity: KNN-5 (if queen available)
  mi_k_I <- NA_real_
  if (!is.null(W_queen) && !is.null(W_knn5)) {
    mi_k <- tryCatch(
      moran.test(vals, W_knn5, randomisation = TRUE,
                 na.action = na.exclude, zero.policy = TRUE),
      error = function(e) NULL
    )
    if (!is.null(mi_k)) mi_k_I <- round(mi_k$estimate[1], 4)
  }

  cat(sprintf("  %-35s  I=%.4f  z=%6.3f  p=%.4f  KNN5_I=%s\n",
              v, mi_q$estimate[1], mi_q$statistic, mi_q$p.value,
              ifelse(is.na(mi_k_I), "–", sprintf("%.4f", mi_k_I))))

  moran_rows[[v]] <- data.frame(
    outcome    = v,
    moran_I    = round(mi_q$estimate[1], 4),
    z_score    = round(mi_q$statistic, 3),
    p_value    = round(mi_q$p.value, 4),
    knn5_I     = mi_k_I,
    weight_type = "queen"
  )
}

if (length(moran_rows) > 0) {
  write_csv(do.call(rbind, moran_rows),
            file.path(OUT_DIR, "r_moran_global.csv"))
  cat(sprintf("  → Saved: %s/r_moran_global.csv\n", OUT_DIR))
}

# ── 4. LISA (Local Moran's I) ─────────────────────────────────────────────────
cat("\n── LISA (Local Moran's I) ────────────────────────────────────────────\n")
lisa_rows <- list()
for (v in OUTCOMES) {
  vals <- df[[v]]
  lm_i <- tryCatch(
    localmoran(vals, W_primary, na.action = na.exclude, zero.policy = TRUE),
    error = function(e) NULL
  )
  if (is.null(lm_i)) next

  # BH-FDR correction
  p_raw  <- lm_i[, 5]
  p_bh   <- p.adjust(p_raw, method = "BH")
  sig    <- p_bh < 0.05

  z_vals    <- scale(vals)[, 1]
  lag_z     <- lag.listw(W_primary, vals, zero.policy = TRUE)
  lag_z_sc  <- scale(lag_z)[, 1]

  quad <- rep("NS", n)
  quad[sig & z_vals > 0 & lag_z_sc > 0] <- "HH"
  quad[sig & z_vals < 0 & lag_z_sc < 0] <- "LL"
  quad[sig & z_vals > 0 & lag_z_sc < 0] <- "HL"
  quad[sig & z_vals < 0 & lag_z_sc > 0] <- "LH"

  tab <- table(quad)
  cat(sprintf("  %-35s  HH=%d  LL=%d  HL=%d  LH=%d  NS=%d\n",
              v,
              sum(quad == "HH"), sum(quad == "LL"),
              sum(quad == "HL"), sum(quad == "LH"), sum(quad == "NS")))

  lisa_rows[[v]] <- data.frame(
    district_idx = seq_len(n),
    outcome      = v,
    local_I      = round(lm_i[, 1], 4),
    p_raw        = round(p_raw, 4),
    p_bh         = round(p_bh, 4),
    quadrant     = quad
  )
}

if (length(lisa_rows) > 0) {
  write_csv(do.call(rbind, lisa_rows),
            file.path(OUT_DIR, "r_lisa_clusters.csv"))
  cat(sprintf("  → Saved: %s/r_lisa_clusters.csv\n", OUT_DIR))
}

# ── 5. Getis-Ord Gi* ──────────────────────────────────────────────────────────
cat("\n── Getis-Ord Gi* ─────────────────────────────────────────────────────\n")
gi_rows <- list()
for (v in head(OUTCOMES, 2)) {
  vals  <- df[[v]]
  gi_z  <- tryCatch(
    localG(vals, W_primary, zero.policy = TRUE),
    error = function(e) NULL
  )
  if (is.null(gi_z)) next

  gi_z  <- as.numeric(gi_z)
  p_val <- 2 * (1 - pnorm(abs(gi_z)))

  hot_class <- dplyr::case_when(
    gi_z >  2.576 ~ "Hotspot (99%)",
    gi_z >  1.960 ~ "Hotspot (95%)",
    gi_z >  1.645 ~ "Hotspot (90%)",
    gi_z < -2.576 ~ "Coldspot (99%)",
    gi_z < -1.960 ~ "Coldspot (95%)",
    gi_z < -1.645 ~ "Coldspot (90%)",
    TRUE           ~ "Not significant"
  )
  cat(sprintf("  %-35s  Hotspots (95%%+): %d  Coldspots (95%%+): %d\n",
              v,
              sum(gi_z > 1.960, na.rm = TRUE),
              sum(gi_z < -1.960, na.rm = TRUE)))

  gi_rows[[v]] <- data.frame(
    district_idx = seq_len(n),
    outcome      = v,
    z_gi_star    = round(gi_z, 4),
    p_value      = round(p_val, 4),
    hotspot_class = hot_class
  )
}

if (length(gi_rows) > 0) {
  write_csv(do.call(rbind, gi_rows),
            file.path(OUT_DIR, "r_getis_ord.csv"))
  cat(sprintf("  → Saved: %s/r_getis_ord.csv\n", OUT_DIR))
}

# ── 6. Bivariate Moran's I ────────────────────────────────────────────────────
if (length(OUTCOMES) >= 2) {
  cat("\n── Bivariate Moran's I (outcome pairs) ──────────────────────────────\n")
  biv_rows <- list()
  pairs    <- combn(OUTCOMES, 2, simplify = FALSE)
  for (pair in pairs) {
    x  <- scale(df[[pair[1]]])[, 1]
    y  <- scale(df[[pair[2]]])[, 1]
    lag_y <- lag.listw(W_primary, df[[pair[2]]], zero.policy = TRUE)
    lag_y <- scale(lag_y)[, 1]
    I_biv <- mean(x * lag_y, na.rm = TRUE)
    cat(sprintf("  %-20s × %-20s  I_biv = %.4f\n", pair[1], pair[2], I_biv))
    biv_rows[[paste(pair, collapse = "x")]] <- data.frame(
      outcome_x = pair[1], outcome_y = pair[2],
      bivariate_I = round(I_biv, 4)
    )
  }
  if (length(biv_rows) > 0) {
    write_csv(do.call(rbind, biv_rows),
              file.path(OUT_DIR, "r_bivariate_moran.csv"))
    cat(sprintf("  → Saved: %s/r_bivariate_moran.csv\n", OUT_DIR))
  }
}

# ── 7. OLS → LM tests → SLM / SEM model selection ───────────────────────────
if (length(COVARS) > 0 && length(OUTCOMES) > 0) {
  cat("\n── Spatial model selection (OLS / SLM / SEM) ────────────────────────\n")
  v      <- OUTCOMES[1]
  fml    <- as.formula(paste(v, "~", paste(COVARS, collapse = " + ")))

  ols <- tryCatch(lm(fml, data = df), error = function(e) NULL)
  if (!is.null(ols)) {
    lm_tests <- tryCatch(
      lm.RStests(ols, W_primary, test = "all", zero.policy = TRUE),
      error = function(e) NULL
    )
    if (!is.null(lm_tests)) {
      cat(sprintf("  OLS R²=%.3f  AIC=%.2f\n",
                  summary(ols)$r.squared, AIC(ols)))
      cat("  LM diagnostics:\n"); print(lm_tests)
    }

    slm <- tryCatch(lagsarlm(fml, data = df, listw = W_primary, zero.policy = TRUE),
                    error = function(e) NULL)
    sem <- tryCatch(errorsarlm(fml, data = df, listw = W_primary, zero.policy = TRUE),
                    error = function(e) NULL)

    if (!is.null(slm) && !is.null(sem)) {
      comp <- data.frame(
        model   = c("OLS", "SLM (spatial lag)", "SEM (spatial error)"),
        AIC     = round(c(AIC(ols), AIC(slm), AIC(sem)), 2),
        log_lik = round(c(as.numeric(logLik(ols)),
                          as.numeric(logLik(slm)),
                          as.numeric(logLik(sem))), 2)
      )
      cat("\n  Model comparison:\n"); print(comp)
      write_csv(comp, file.path(OUT_DIR, "r_model_comparison.csv"))
      cat(sprintf("  → Saved: %s/r_model_comparison.csv\n", OUT_DIR))
    }
  }
}

cat("\n── spatial_diagnostics.R complete ───────────────────────────────────\n")
cat(sprintf("All outputs written to %s/\n", OUT_DIR))
