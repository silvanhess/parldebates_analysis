# library(haven)
library(tidyverse)
library(httr)

# Step 1: Datensatz
df <- readRDS("Data/labeled_dataset_cleaned.rds") |> 
  filter(final_climate == 1)

# Step 2: API-Key
APIkey <- trimws(readLines("openai_key.txt", warn = FALSE))

# Step 3: Prompt
prompt <- paste(readLines("extraction_prompt.txt"), collapse = " ")

# === SET NUMBER OF CASES HERE ===
n_cases <- NULL # Anzahl Fälle, die durchlaufen sollen (NULL für alle)
# ================================

# Initialize new column
df$keywords <- NA_character_

# # Initialize mini dataset
# mini_data <- data.frame(
#   paragraph_id = character(),
#   original_text = character(),
#   keywords = character(),
#   stringsAsFactors = FALSE
# )

# Counter for processed cases
processed <- 0

# API Loop
for (i in seq_len(nrow(df))) {
  # Stop if we've reached the limit
  if (!is.null(n_cases) && processed >= n_cases) {
    cat("Reached limit of", n_cases, "cases.\n")
    break
  }

  # id <- df$paragrph_id[i]
  text <- df$original_text[i]

  # # Skip if empty
  # if (is.na(title) || title == "") {
  #   next
  # }

  r <- httr::POST(
    url = "https://api.openai.com/v1/chat/completions",
    content_type("application/json"),
    add_headers(Authorization = paste("Bearer", APIkey)),
    body = list(
      model = "gpt-5-mini",
      messages = list(
        list(role = "system", content = prompt),
        list(role = "user", content = text)
      )
    ),
    encode = "json"
  )

  result <- tryCatch(
    {
      c <- httr::content(r, as = "parsed", type = "application/json")
      if (!is.null(c$choices[[1]]$message$content)) {
        c$choices[[1]]$message$content
      } else {
        NA_character_
      }
    },
    error = function(e) NA_character_
  )

  # Add to df
  df$keywords[i] <- result

  # # Add to mini dataset
  # mini_data <- rbind(
  #   mini_data,
  #   data.frame(
  #     BusinessShortNumber = bn,
  #     Title = title,
  #     class = result,
  #     stringsAsFactors = FALSE
  #   )
  # )

  processed <- processed + 1
  cat("Finished case", processed, "/", n_cases, ":", result, "\n")
}

# Save the updated dataframe
write_rds(df, "Data/keywords_extracted.rds")
