# Function to translate text using DeepL API
deepl_translate <- function(
  text,
  auth_key = Sys.getenv("DEEPL_API_KEY")
) {
  require(httr)
  require(jsonlite)

  url <- "https://api.deepl.com/v2/translate"

  response <- POST(
    url,
    add_headers(Authorization = paste("DeepL-Auth-Key", auth_key)),
    body = list(
      text = text,
      target_lang = "EN"
    ),
    encode = "form"
  )

  # error handling
  if (httr::status_code(response) != 200) {
    stop(
      "DeepL API Fehler: ",
      httr::status_code(response),
      " - ",
      httr::content(response, "text", encoding = "UTF-8")
    )
  }

  # parse
  result <- httr::content(response, as = "parsed")

  # extract text
  result$translations[[1]]$text
}

# Histogram for Word Count Distribution
plot_word_count_distribution <- function(
  data,
  word_count_col = "paragraph_word_count",
  max_limit = 300,
  # vline_position = 256,
  # vline_label = "Max Tokens",
  x_label = "Paragraph Length (in words)",
  y_label = "Number of Paragraphs",
  fill_color = "steelblue",
  output_path = NULL
) {
  require(ggplot2)
  require(here)

  p <- ggplot(data, aes(x = .data[[word_count_col]])) +
    geom_histogram(fill = fill_color) +
    xlim(0, max_limit) +
    # geom_vline(
    #   xintercept = vline_position,
    #   color = "red",
    #   linetype = "dashed"
    # ) +
    # annotate(
    #   "text",
    #   x = vline_position + 4,
    #   y = annotation_y,
    #   label = vline_label,
    #   color = "red",
    #   hjust = 1.5
    # ) +
    theme_minimal() +
    labs(
      x = x_label,
      y = y_label
    )

  if (!is.null(output_path)) {
    ggsave(here(output_path), plot = p)
    write_rds(p, here(output_path %>% str_replace(".png", ".rds")))
  }

  return(p)
}


# Barplot for categorical distribution
plot_categorical_distribution <- function(
  data,
  group_col,
  x_label = NULL,
  y_label = "Percentage of Paragraphs",
  label_suffix = "paragraphs",
  fill_color = "steelblue",
  text_color = "black",
  output_path = NULL
) {
  require(ggplot2)
  require(dplyr)
  require(scales)
  require(here)

  df_grouped <- data |>
    group_by(across(all_of(group_col))) |>
    summarise(
      group_count = n(),
      pct = group_count / nrow(data),
      .groups = "drop"
    )

  # Standard x_label falls nicht angegeben
  if (is.null(x_label)) {
    x_label <- group_col
  }

  p <- ggplot(df_grouped, aes(x = .data[[group_col]], y = pct)) +
    geom_col(fill = fill_color) +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      x = x_label,
      y = y_label
    ) +
    theme_minimal() #+
  # geom_text(
  #   aes(label = paste(group_count, label_suffix)),
  #   vjust = 1.5,
  #   color = text_color
  # )

  if (!is.null(output_path)) {
    ggsave(here(output_path), plot = p)
    write_rds(p, here(output_path %>% str_replace(".png", ".rds")))
  }

  return(p)
}
