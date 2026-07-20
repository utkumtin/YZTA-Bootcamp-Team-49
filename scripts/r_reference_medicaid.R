# S2-14 — R referans fixture'ı (fixest::feols).
# pyfixest OLS/TWFE tolerans-doğrulaması için referans sonuçlar üretir.
#
# Girdi : runs/r_fixture/medicaid_panel.csv  (önce: .venv/bin/python scripts/export_r_panel.py)
# Çıktı : tests/fixtures/r_reference_medicaid.csv
# Koşum : RStudio'da bu dosyayı aktif sekmede Source et YA DA `Rscript scripts/r_reference_medicaid.R`
#
# Spec matrisi pareto/analysis/estimators.py ile birebir aynı:
#   formül  : outcome ~ treated_post + kontroller  (TWFE: | county_fips + year)
#   vcov    : cluster-robust, state_fips (pyfixest CRV1 = fixest varsayılan ssc)
#   NA      : fit kolonları üzerinden explicit complete.cases (pyfixest dropna eşleniği)

locate_repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1) {
    return(dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (nzchar(p)) return(dirname(dirname(normalizePath(p))))
  }
  stop("Repo kökü bulunamadı: Rscript ile çalıştırın ya da script'i RStudio'da aktif sekmede Source edin.")
}

repo_root <- locate_repo_root()
setwd(repo_root)

if (!requireNamespace("fixest", quietly = TRUE)) {
  install.packages("fixest", repos = "https://cloud.r-project.org")
}
library(fixest)

panel_path <- file.path("runs", "r_fixture", "medicaid_panel.csv")
if (!file.exists(panel_path)) {
  stop("Panel CSV yok — önce scripts/export_r_panel.py çalıştırın: ", panel_path)
}
panel <- read.csv(
  panel_path,
  colClasses = c(county_fips = "character", state_fips = "character")
)

CONTROLS <- c("median_hh_income", "poverty_rate", "unemployment_rate")

specs <- expand.grid(
  outcome = c("crude_rate", "pct_uninsured"),
  estimator = c("ols", "twfe"),
  with_controls = c(FALSE, TRUE),
  weighted = c(FALSE, TRUE),
  stringsAsFactors = FALSE
)

rows <- lapply(seq_len(nrow(specs)), function(i) {
  s <- specs[i, ]
  controls <- if (s$with_controls) CONTROLS else character(0)
  fit_cols <- c(s$outcome, "treated_post", controls, "state_fips")
  if (s$weighted) fit_cols <- c(fit_cols, "population")
  sub <- panel[complete.cases(panel[, fit_cols]), ]

  rhs <- paste(c("treated_post", controls), collapse = " + ")
  fml_txt <- paste0(s$outcome, " ~ ", rhs)
  if (s$estimator == "twfe") fml_txt <- paste0(fml_txt, " | county_fips + year")

  fit <- if (s$weighted) {
    feols(as.formula(fml_txt), data = sub, cluster = ~state_fips, weights = ~population)
  } else {
    feols(as.formula(fml_txt), data = sub, cluster = ~state_fips)
  }

  ct <- coeftable(fit)["treated_post", ]
  ci <- confint(fit)["treated_post", ]
  data.frame(
    spec_id = paste0(
      s$estimator, "|", s$outcome,
      "|controls=", as.integer(s$with_controls),
      "|weights=", as.integer(s$weighted)
    ),
    estimator = s$estimator,
    outcome = s$outcome,
    controls = paste(controls, collapse = ";"),
    weight_col = if (s$weighted) "population" else "",
    cluster_by = "state_fips",
    coefficient = unname(ct["Estimate"]),
    std_error = unname(ct["Std. Error"]),
    p_value = unname(ct[4]),
    ci_low = unname(ci[[1]]),
    ci_high = unname(ci[[2]]),
    # nobs(fit) = singleton-FE düşümü SONRASI; pyfixest tarafı fit-öncesi
    # complete-cases sayısını raporlar → test n_complete_cases ile eşleşmeli.
    n_obs = nobs(fit),
    n_complete_cases = nrow(sub),
    stringsAsFactors = FALSE
  )
})

out <- do.call(rbind, rows)
out$r_version <- paste(R.version$major, R.version$minor, sep = ".")
out$fixest_version <- as.character(packageVersion("fixest"))

out_path <- file.path("tests", "fixtures", "r_reference_medicaid.csv")
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write.csv(out, out_path, row.names = FALSE)

cat("R fixture yazıldı:", out_path, "-", nrow(out), "spec\n")
print(out[, c("spec_id", "coefficient", "std_error", "p_value", "n_obs")])
