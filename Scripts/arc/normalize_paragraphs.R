# normalize paragraph lengths --------------------------------------------

# paragraph length needs to be between 100 and 1000 characters

# paragraphs shorter than 100 characters are appended to the previous paragraph
# or the next paragraph if they are the first paragraph in the text

# paragraphs longer than 1000 characters are split into smaller paragraphs of max 1000 characters
# to split the paragraphs, we split after the last full stop before the 1000 character limit

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")

transcripts_normalized <- transcripts_cleaned |>
  arrange(ID, paragraph_id) |>
  group_by(ID) |>
  mutate(
    paragraph = {
      # append short paragraphs to previous or next paragraph
      paragraphs <- paragraph
      for (i in seq_along(paragraphs)) {
        if (nchar(paragraphs[i]) < 100) {
          if (i == 1) {
            paragraphs[i + 1] <- paste(paragraphs[i], paragraphs[i + 1])
            paragraphs[i] <- NA
          } else {
            paragraphs[i - 1] <- paste(paragraphs[i - 1], paragraphs[i])
            paragraphs[i] <- NA
          }
        }
      }
      paragraphs <- na.omit(paragraphs)
      
      # split long paragraphs into smaller chunks
      new_paragraphs <- c()
      for (para in paragraphs) {
        while (nchar(para) > 1000) {
          split_pos <- str_locate_all(para, "\\. ")[[1]]
          split_pos <- split_pos[split_pos[,1] <= 1000, , drop = FALSE]
          if (nrow(split_pos) == 0) {
            split_index <- 1000
          } else {
            split_index <- max(split_pos[,1]) + 1
          }
          new_paragraphs <- c(new_paragraphs, str_sub(para, 1, split_index))
          para <- str_sub(para, split_index + 1)
        }
        new_paragraphs <- c(new_paragraphs, para)
      }
      new_paragraphs
    }
  ) |>
  ungroup() |>
  filter(!is.na(paragraph)) |>
  mutate(
    Textlength = nchar(paragraph)
  )