import pandas as pd
from bertopic import BERTopic
import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# 1. Read in CSV
df = pd.read_csv('Data/transcripts_climate_for_topic_modeling.csv')
text_column = 'transcript_text'

# Extract the text data
documents = df[text_column].tolist()

# 2. Initialize BERTopic model
# Configuration optimized for ~6000 documents with ~500 words each
# Targeting 10-30 topics with multilingual support for French and German

# Configure UMAP with balanced settings
umap_model = UMAP(
    n_neighbors=15,  # Default value for balanced granularity
    n_components=5,  # Dimensionality of reduced space
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# Configure HDBSCAN with more granular clustering
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=30,  # Reduced to allow more topics
    min_samples=3,  # Lower value allows more clusters to form
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# Configure CountVectorizer to filter rare and very common words
vectorizer_model = CountVectorizer(
    min_df=2,  # Ignore words that appear in fewer than 2 documents
    max_df=0.95,  # Ignore words that appear in more than 95% of documents
    ngram_range=(1, 2)  # Include both single words and bigrams
)

topic_model = BERTopic(
    language="multilingual",
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,  # Filter rare/common words
    nr_topics="auto",  # Allows automatic merging if too many topics
    verbose=True
)

# 3. Fit the model and transform documents
topics, probs = topic_model.fit_transform(documents)

# Add topics back to the dataframe
df['topic'] = topics
df['topic_probability'] = probs

# Display basic information
print(f"\nNumber of topics found: {len(set(topics)) - 1}")  # -1 for outlier topic
print("\nTopic distribution:")
print(df['topic'].value_counts().head(10))

# Get topic information
topic_info = topic_model.get_topic_info()
print("\nTopic Information:")
print(topic_info.head(10))

# Save results
df.to_csv('Data/documents_with_topics.csv', index=False)
topic_info.to_csv('Data/topic_info.csv', index=False)

# Optional: Visualize topics (uncomment if needed)
fig = topic_model.visualize_topics()
fig.write_html("Outputs/topic_visualization.html")

# Optional: Get representative documents for a specific topic
# topic_num = 0  # Change to the topic you want to explore
# print(f"\nRepresentative documents for Topic {topic_num}:")
# print(topic_model.get_representative_docs(topic_num))

print("\nProcessing complete!")