library(httr)
library(jsonlite)

deepl_translate <- function(text, auth_key = Sys.getenv("DEEPL_API_KEY")) {
  url <- "https://api.deepl.com/v2/translate"

  response <- POST(
    url,
    body = list(
      auth_key = auth_key,
      text = text,
      target_lang = "EN"
    ),
    encode = "form"
  )

  # error handling
  if (response$status_code != 200) {
    stop(
      "DeepL API Fehler: ",
      status_code(response),
      " - ",
      content(response, "text")
    )
  }

  # parse
  result <- content(response, as = "parsed")

  # extract text
  result$translations[[1]]$text
}
