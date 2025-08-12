# Create a timestamp string
timestamp <- format(Sys.time(), "%Y%m%d-%H%M%S")
pdf_file <- paste0("domain_vs_rays", "_", timestamp, ".pdf")

pdf(pdf_file, width = 8, height = 6)
par(mar = c(5, 4, 5, 2))  # Extra space at the top for 2nd x-axis

single_ray_estimate = 712
ray_estimate <- function(x, ray_size_bytes = single_ray_estimate) {
  (x * ray_size_bytes) / (1024 * 1024 * 1024) # Bytes to GiB
}

domain_estimate <- function(x, y, z) {
  (x / 1024) * (y / 1024) * (z / 1024) * (32 / 8)
}

x1 <- 10^seq(0, 9, length.out = 1000)
y1 <- ray_estimate(x1)

x2 <- 2^seq(0, log(4096 + 2048, base = 2), length.out = 64)
y2 <- domain_estimate(x2, x2, x2)

# Main plot with base-10 log x-axis
plot(x1, y1, type = "l", col = "blue", lwd = 2,
     xlab = "Number of rays (Np)", ylab = "Est. memory usage (GiB)",
     main = "Domain size versus number of rays relevance to estimated memory usage",
     ylim = range(c(y1, y2)),
     log = "x"  # base-10 log scale on x-axis
)

# Additional ray lines (same x scale, so log scale is automatic)
y1_512 <- y1 + domain_estimate(512, 512, 512)
lines(x1, y1_512, col = "red", lwd = 2, lty = 2)

y1_1024 <- y1 + domain_estimate(1024, 1024, 1024)
lines(x1, y1_1024, col = "purple", lwd = 2, lty = 2)

y1_2048 <- y1 + domain_estimate(2048, 2048, 2048)
lines(x1, y1_2048, col = "orange", lwd = 2, lty = 2)

y1_4096 <- y1 + domain_estimate(4096, 4096, 4096)  # fixed: was domain_estimate(2048) before
lines(x1, y1_4096, col = "magenta", lwd = 2, lty = 2)

batcher_results <- function(dims, Np, mem_available_gb = 40, leeway_factor = 1.1, ray_size_bytes = single_ray_estimate, allocation_count = 4) {
  mem_available = mem_available_gb * (1024 * 1024 * 1024)

  predicted_domain_allocation <- domain_estimate(dims, dims, dims) * (1024 * 1024 * 1024)
  ray_memory_raw <- (ray_size_bytes * Np) / (1024 * 1024 * 1024)
  print(ray_memory_raw)

  # Total estimate with leeway factor
  limiting_value <- (predicted_domain_allocation * allocation_count + ray_memory_raw) * leeway_factor
  print(limiting_value)

  # Determine ray batches needed to fit memory available
  ray_batch_count <- max(1, ceiling(ray_memory_raw * leeway_factor / mem_available))
  print(ray_batch_count)
  # Determine domain regions needed to fit per batch memory limit
  domain_region_count <- max(1, ceiling((limiting_value - (ray_memory_raw / ray_batch_count)) / mem_available))
  print(domain_region_count)

  return(domain_estimate(dims, dims, dims %/% domain_region_count) + ray_estimate(Np %/% ray_batch_count, ray_size_bytes = ray_size_bytes))
}

lines(x1, batcher_results(1024, x1), col = "pink", lwd = 2, lty = 2)

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
       legend = c("Ray estimate",
                  "Total est. in a 512 domain", "Total est. in a 1024 domain", "Total est. in a 2048 domain", "Total est. in a 4096 domain",
                  "Batched est. in a 1024 domain", "Domain estimate"),
       col = c("blue", "red", "purple", "orange", "magenta", "pink", "darkgreen"),
       lty = c(1, 2, 2, 2, 2, 1), lwd = 2)

dev.off()