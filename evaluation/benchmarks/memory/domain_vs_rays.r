# Create a timestamp string
timestamp <- format(Sys.time(), "%Y%m%d-%H%M%S")
pdf_file <- paste0("domain_vs_rays", "_", timestamp, ".pdf")

pdf(pdf_file, width = 8, height = 6)
par(mar = c(5, 4, 5, 2))  # Extra space at the top for 2nd x-axis

ray_estimate <- function(x) {
  (x * 712) / (1024 * 1024 * 1024) # Bytes to GiB
}

domain_estimate <- function(x) {
  (x / 1024)^3 * (32 / 8)
}

x1 <- 10^seq(0, 9, length.out = 1000)
y1 <- ray_estimate(x1)

x2 <- 2^seq(0, log(4096 + 2048, base = 2), length.out = 64)
y2 <- domain_estimate(x2)

# Main plot with base-10 log x-axis
plot(x1, y1, type = "l", col = "blue", lwd = 2,
     xlab = "Number of rays (Np)", ylab = "Est. memory usage (GiB)",
     main = "Domain size versus number of rays relevance to estimated memory usage",
     ylim = range(c(y1, y2)),
     log = "x"  # base-10 log scale on x-axis
)

# Additional ray lines (same x scale, so log scale is automatic)
y1_512 <- y1 + domain_estimate(512)
lines(x1, y1_512, col = "red", lwd = 2, lty = 2)

y1_1024 <- y1 + domain_estimate(1024)
lines(x1, y1_1024, col = "purple", lwd = 2, lty = 2)

y1_2048 <- y1 + domain_estimate(2048)
lines(x1, y1_2048, col = "orange", lwd = 2, lty = 2)

y1_4096 <- y1 + domain_estimate(4096)  # fixed: was domain_estimate(2048) before
lines(x1, y1_4096, col = "magenta", lwd = 2, lty = 2)

# Overlay second plot (domain estimate) without axes
par(new = TRUE)

plot(x2, y2, type = "l", col = "darkgreen", lwd = 2,
     axes = FALSE, xlab = "", ylab = "", main = "",
     log = "x",  # Important! match log10 scale of bottom axis
     xlim = range(x1),  # ensure x ranges match exactly
     ylim = range(c(y1, y2))  # same y limits
)

axis(side = 3, at = pretty(x2), labels = pretty(x2), col = "darkgreen", col.axis = "darkgreen")
mtext("Cubic resolution of domain", side = 3, line = 3, col = "darkgreen")

legend("topleft",
       legend = c("Ray estimate", "Ray est. in a 512 domain", "Ray est. in a 1024 domain",
                  "Ray est. in a 2048 domain", "Ray est. in a 4096 domain", "Domain estimate"),
       col = c("blue", "red", "purple", "orange", "magenta", "darkgreen"),
       lty = c(1, 2, 2, 2, 2, 1), lwd = 2)

dev.off()