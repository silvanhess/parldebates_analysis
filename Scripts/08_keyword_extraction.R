# Load necessary libraries
library(tidyverse)
library(dotenv)
library(ellmer)
library(furrr)
library(future)

# Step 1: Load the dataset
df <- readRDS("Data/labeled_dataset_cleaned.rds") |>
  filter(final_climate == 1)

# Initialize keywords column
df$keywords <- NA_character_

# Step 2: Load API-Key from .env
load_dot_env()

# Step 3: Prepare the prompt
prompt <- paste(readLines("extraction_prompt.txt"), collapse = " ")

# Set up parallel processing
# Adjust workers based on your API rate limits and system capacity
plan(multisession, workers = 4)

# Function to process a single text entry
process_text <- function(text, index) {
  # Initialize chat inside the function for parallel processing
  chat <- chat_openai(
    model = "gpt-5-mini",
    system_prompt = prompt
  )
  
  # Skip if empty
  if (is.na(text) || text == "") {
    message("Skipping empty row ", index)
    return(NA_character_)
  }

  # Get the response from the chat model
  result <- tryCatch(
    {
      response <- chat$chat(text)
      if (!is.null(response)) {
        message("Successfully processed row ", index)
        response
      } else {
        message("No response for row ", index)
        NA_character_
      }
    },
    error = function(e) {
      message("Error processing row ", index, ": ", e$message)
      NA_character_
    }
  )

  return(result)
}

# Process all rows
rows_to_process <- seq_len(nrow(df))

cat("Total rows to process:", length(rows_to_process), "\n")
cat("Using parallel processing with", future::nbrOfWorkers(), "workers\n")

# Process texts in parallel and store results
df$keywords <- future_map2_chr(
  df$original_text,
  rows_to_process,
  process_text,
  .progress = TRUE,
  .options = furrr_options(seed = TRUE)
)

# Close parallel workers
plan(sequential)

# Save the updated dataframe
write_rds(df, "Data/keywords_extracted.rds")
