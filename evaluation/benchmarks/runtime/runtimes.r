# Load CSV
data <- read.csv("benchmark_results.csv")

# Install ggplot2 if not already installed
if (!require(ggplot2)) {
  install.packages("ggplot2")
  library(ggplot2)
} else {
  library(ggplot2)
}

# Plot: Rays vs Time with dim as group
ggplot(data, aes(x = rays, y = time, color = as.factor(dim))) +
  geom_line() +
  geom_point() +
  labs(
    title = "Time vs Rays for Different Dimensions",
    x = "Number of Rays",
    y = "Time (s)",
    color = "Dimension"
  ) +
  theme_minimal()