# Install ggplot2 if not already installed
if (!require(ggplot2)) {
  install.packages("ggplot2", lib = Sys.getenv("R_LIBS_USER"), repos = "https://cloud.r-project.org")
  library(ggplot2)
} else {
  library(ggplot2)
}

# Get command-line arguments (skip the first which is the script name)
args <- commandArgs(trailingOnly = TRUE)

# Check if file argument is provided
if (length(args) < 1) {
  stop("Usage: Rscript plot_benchmark.R <benchmark_csv_file>")
}

# Get the CSV filename from command line
csv_file <- args[1]

# Load CSV
data <- read.csv(csv_file)

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