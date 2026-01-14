import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from transformers import pipeline
from nltk.corpus import stopwords
import nltk
from pathlib import Path
import pickle

# 1. Read in CSV with preprocessed data
df = pd.read_csv('Data/transcripts_climate_for_topic_modeling.csv')
text_column = 'transcript_text'
documents = df[text_column].tolist()

# 2. Pre-calculate embeddings (load if available otherwise compute and save)
print("Calculating or loading embeddings...")
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

emb_path = Path('Data/embeddings.pickle')
if emb_path.exists():
    with emb_path.open('rb') as handle:
        embeddings = pickle.load(handle)
else:
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    with emb_path.open('wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 3. Pre-reduce embeddings for visualization (2D)
print("Reducing embeddings for visualization...")
reduced_embeddings = UMAP(
    n_neighbors=15, 
    n_components=2, 
    min_dist=0.0, 
    metric='cosine', 
    random_state=42
).fit_transform(embeddings)

# 4. UMAP model for topic modeling (5D)
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# 5. HDBSCAN clustering model
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=50,  # ~1.7% of 6000 docs
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# 6. CountVectorizer with stopwords

# Download stopwords (only needed once)
nltk.download('stopwords', quiet=True)

# Get German and French stopwords
german_stopwords = set(stopwords.words('german'))
french_stopwords = set(stopwords.words('french'))
all_stopwords = german_stopwords.union(french_stopwords)

# Add Swiss German specific stopwords
swiss_german_stopwords = {'dass'}  # Add more Swiss German words here if needed
all_stopwords = all_stopwords.union(swiss_german_stopwords)

vectorizer_model = CountVectorizer(
    min_df=2,
    max_df=0.85,
    ngram_range=(1, 2),
    stop_words=list(all_stopwords)
)

# 7. Representation models for better topic labels
keybert = KeyBERTInspired()
mmr = MaximalMarginalRelevance(diversity=0.3)

# Flan-T5 for topic label generation (works well on CPU)
print("Loading Flan-T5 model for topic representation...")
generator = pipeline('text2text-generation', model='google/flan-t5-large', device=-1)

# Prompt optimized for Flan-T5
prompt = """
I have a topic described by the following keywords: [KEYWORDS]

The topic contains these representative documents:
[DOCUMENTS]

Based on the keywords and documents above, generate a short, descriptive label for this topic (maximum 5 words). Only return the label itself, nothing else.

Topic label:"""

flan = TextGeneration(generator, prompt=prompt)

representation_model = {
    "KeyBERT": keybert,
    "MMR": mmr,
    "Flan-T5": flan,
}

# 8. Initialize and fit BERTopic
print("Fitting BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    nr_topics="auto",
    verbose=True
)

topics, probs = topic_model.fit_transform(documents, embeddings)

# 9. Add topics to dataframe
df['topic'] = topics
df['topic_probability'] = probs

# 10. Display results
print(f"\nNumber of topics found: {len(set(topics)) - 1}")  # -1 for outlier topic
print("\nTopic distribution:")
print(df['topic'].value_counts())

# 11. Get topic information with multiple representations
topic_info = topic_model.get_topic_info()
print("\nTopic Information:")
print(topic_info)

# 12. Save results
df.to_csv('Data/documents_with_topics.csv', index=False)
topic_info.to_csv('Data/topic_info.csv', index=False)

# 13. Create visualizations
print("\nCreating visualizations...")

# Visualize documents with pre-reduced embeddings
fig_docs = topic_model.visualize_documents(
    documents, 
    reduced_embeddings=reduced_embeddings,
    hide_document_hover=False,
    hide_annotations=False
)
fig_docs.write_html("Outputs/topic_documents_visualization.html")

# Visualize intertopic distance map
fig_topics = topic_model.visualize_topics()
fig_topics.write_html("Outputs/intertopic_distance_map.html")

# Visualize topic hierarchy
fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html("Outputs/topic_hierarchy.html")

# Visualize barchart of top words per topic
fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
fig_barchart.write_html("Outputs/topic_barchart.html")

print("\nProcessing complete!")