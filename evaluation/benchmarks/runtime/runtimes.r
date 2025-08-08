# Install ggplot2 & tidyr if not already installed
if (!require(ggplot2)) install.packages("ggplot2", lib = Sys.getenv("R_LIBS_USER"), repos = "https://cloud.r-project.org")
if (!require(tidyr)) install.packages("tidyr", repos = "https://cloud.r-project.org")

# Load libraries
library(ggplot2)
library(tidyr)
# contains general tools, built-in library
# if libraries aren't imported they can be referenced as, library::function, I think? test later if curious.
library(tools)

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

#Use pivot_longer() to stack runtime and legacyRuntime into a single column
data_long <- pivot_longer(
  data,
  cols = c(runtime, legacyRuntime),
  names_to = "type",
  values_to = "time"
)

# <- represents assignment, don't have to have this, will save file anyway, but it would be under the default name "Rplots.pdf"
# by assigning to a variable we can save it ourselves under a different name
result <- ggplot(data_long, aes(x = rays, y = time, color = factor(dims), linetype = type, shape = type)) +
  geom_line(linewidth = 1) + #, arrow = arrow(type = "closed", length = unit(0.15, "inches"))) +
  geom_point(size = 3) +
  scale_linetype_manual(values = c(runtime = "solid", legacyRuntime = "dashed")) +
  scale_shape_manual(values = c(runtime = 16, legacyRuntime = 4)) +  # 16=solid circle, 17=triangle
  labs(
    title = "Execution time vs Np for various resolution cubic simulations (Legacy vs Updated)",
    x = "Number of Rays (Np)",
    y = "Time (s)",
    color = "Dimension",
    linetype = "Runtime Type",
    shape = "Runtime Type"
  ) +
  theme_minimal()

# sanitises csv_file to remove extension
base_name <- file_path_sans_ext(csv_file)
# adds the correct extension on so we can save with the same name as import but (obviously) a different format
pdf_file <- paste0(base_name, ".pdf")

ggsave(filename = pdf_file, plot = result, width = 8, height = 6, units = "in", dpi = 300)