library(haven)
library(dplyr)
library(httr)
library(stringr)

# Step 1: Datensatz
df <- readRDS("Data/businesses_cleaned.rds") |>
  select(BusinessShortNumber, Title, BusinessDetails)

# Step 2: API-Key
APIkey <- trimws(readLines("openai_key.txt", warn = FALSE))

# Step 3: Prompt
prompt <- paste(readLines("prompt.txt"), collapse = " ")

# === SET NUMBER OF CASES HERE ===
n_cases <- 1000 # Anzahl Fälle, die durchlaufen sollen (NULL für alle)
# ================================

# Initialize new column
df$class <- NA_character_

# Initialize mini dataset
mini_data <- data.frame(
  BusinessShortNumber = character(),
  Title = character(),
  class = character(),
  stringsAsFactors = FALSE
)

# Counter for processed cases
processed <- 0

# API Loop - only for rows with non-empty open answers
for (i in seq_len(nrow(df))) {
  # Stop if we've reached the limit
  if (!is.null(n_cases) && processed >= n_cases) {
    cat("Reached limit of", n_cases, "cases.\n")
    break
  }

  bn <- df$BusinessShortNumber[i]
  title <- df$Title[i]

  # Skip if empty
  if (is.na(title) || title == "") {
    next
  }

  r <- httr::POST(
    url = "https://api.openai.com/v1/chat/completions",
    content_type("application/json"),
    add_headers(Authorization = paste("Bearer", APIkey)),
    body = list(
      model = "gpt-5-mini",
      messages = list(
        list(role = "system", content = prompt),
        list(role = "user", content = title)
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
  df$class[i] <- result

  # Add to mini dataset
  mini_data <- rbind(
    mini_data,
    data.frame(
      BusinessShortNumber = bn,
      Title = title,
      class = result,
      stringsAsFactors = FALSE
    )
  )

  processed <- processed + 1
  cat("Finished case", processed, "/", n_cases, ":", result, "\n")
}
