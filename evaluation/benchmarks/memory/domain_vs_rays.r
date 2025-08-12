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

batcher_results <- function(dims, Np_vec, mem_available_gb = 40, leeway_factor = 1.1, ray_size_bytes = 712, allocation_count = 4) {
  # Convert memory available to bytes
  mem_available <- mem_available_gb * (1024^3)

  # Predicted domain allocation (in bytes)
  predicted_domain_allocation <- domain_estimate(dims, dims, dims) * (1024^3)

  # Allocate result vector
  results <- numeric(length(Np_vec))

  for (i in seq_along(Np_vec)) {
    Np <- Np_vec[i]

    ray_memory_raw <- ray_size_bytes * Np

    limiting_value <- (predicted_domain_allocation * allocation_count + ray_memory_raw) * leeway_factor

    ray_batch_count <- max(1, ceiling(ray_memory_raw * leeway_factor / mem_available))
    print("RBC")
    print(ray_batch_count)
    domain_region_count <- max(1, ceiling((limiting_value - (ray_memory_raw / ray_batch_count)) / mem_available))
    print("DRC")
    print(domain_region_count)

    # Store result in GiB
    results[i] <- domain_estimate(dims, dims, dims %/% domain_region_count) + ray_estimate(Np %/% ray_batch_count, ray_size_bytes)
  }

  return(results)
}

x1 <- 10^seq(0, 9, length.out = 1000)
y1 <- ray_estimate(x1)

x2 <- 2^seq(0, log(4096 + 2048, base = 2), length.out = 64)
y2 <- domain_estimate(x2, x2, x2)

ylim_range <- range(c(y1, y2))
ylim_capped <- c(ylim_range[1], min(ylim_range[-1], 64))

# Main plot with base-10 log x-axis
plot(x1, y1, type = "l", col = "blue", lwd = 2,
     xlab = "Number of rays (Np)", ylab = "Est. memory usage (GiB)",
     main = "Domain size versus number of rays relevance to estimated memory usage",
     ylim = ylim_capped,
     log = "x" # base-10 log scale on x-axis
)

# Additional ray lines (same x scale, so log scale is automatic)
y1_1024 <- y1 + domain_estimate(1024, 1024, 1024)
lines(x1, y1_1024, col = "purple", lwd = 2, lty = 2)
lines(x1, batcher_results(1024, x1), col = "purple", lwd = 2, lty = 3)

y1_1536 <- y1 + domain_estimate(1536, 1536, 1536)
lines(x1, y1_1536, col = "orange", lwd = 2, lty = 2)
lines(x1, batcher_results(1536, x1), col = "orange", lwd = 2, lty = 3)

y1_2048 <- y1 + domain_estimate(2048, 2048, 2048)
lines(x1, y1_2048, col = "magenta", lwd = 2, lty = 2)
lines(x1, batcher_results(2048, x1), col = "magenta", lwd = 2, lty = 3)

# Overlay second plot (domain estimate) without axes
par(new = TRUE)

plot(x2, y2, type = "l", col = "darkgreen", lwd = 2,
     axes = FALSE, xlab = "", ylab = "", main = "",
     log = "x",  # Important! match log10 scale of bottom axis
     xlim = range(x1),  # ensure x ranges match exactly
     ylim = ylim_capped  # same y limits
)

abline(h = 40, col = "black", lwd = 2, lty = 1)

axis(side = 3, at = pretty(x2), labels = pretty(x2), col = "darkgreen", col.axis = "darkgreen")
mtext("Cubic resolution of domain", side = 3, line = 3, col = "darkgreen")

legend("topleft",
       legend = c("Ray estimate",
                  "Total est. in a 1024 domain", "Batched est. in a 1024 domain",
                  "Total est. in a 1536 domain", "Batched est. in a 1536 domain",
                  "Total est. in a 2048 domain", "Batched est. in a 2048 domain",
                  "Domain estimate"),
       col = c("blue", "purple", "purple", "orange", "orange", "magenta", "magenta", "darkgreen"),
       lty = c(1, 2, 3, 2, 3, 2, 3, 1), lwd = 2)

dev.off()