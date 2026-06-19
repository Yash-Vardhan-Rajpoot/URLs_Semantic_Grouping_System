import streamlit as st
import pandas as pd

from src.scraper import scrape_urls
from src.embedding import generate_embeddings, generate_similarity_matrix
from src.clustering import perform_clustering
from src.category import create_result_dataframe

st.set_page_config(
    page_title="URL Semantic Grouping System",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("URL Semantic Grouping")
st.sidebar.write(
    "Paste each URL on its own line, then click Generate Clusters to group "
    "similar pages by semantic content."
)

url_text = st.sidebar.text_area(
    "Enter URLs",
    value="https://example.com\nhttps://en.wikipedia.org/wiki/Natural_language_processing",
    height=240,
)

run_button = st.sidebar.button("Generate Clusters")

st.title("🔗 URL Semantic Grouping System")
st.subheader("Semantic URL Clustering using NLP Embeddings")
st.markdown(
    """
This app extracts page content from URLs, converts the text to embeddings, and groups
similar URLs into categories. Use the sidebar to enter URLs and run the clustering workflow.
"""
)

if run_button:
    urls = [url.strip() for url in url_text.split("\n") if url.strip()]

    if len(urls) < 2:
        st.error("Please enter at least 2 URLs.")
    else:
        with st.spinner("Extracting content from URLs..."):
            documents = scrape_urls(urls)
            documents = [doc[:5000] for doc in documents]

        with st.spinner("Generating embeddings..."):
            embeddings = generate_embeddings(documents)

        with st.spinner("Calculating similarity matrix..."):
            sim_matrix = generate_similarity_matrix(embeddings)

        with st.spinner("Clustering URLs..."):
            labels, threshold = perform_clustering(embeddings, sim_matrix)

        result = create_result_dataframe(urls, labels)
        category_count = result["Category"].nunique()

        st.success(f"Found {category_count} categories")
        st.metric("URLs processed", len(urls))
        st.metric("Categories created", category_count)

        st.subheader("Cluster Results")
        st.dataframe(result, use_container_width=True)

        st.subheader("Similarity Matrix")
        sim_df = pd.DataFrame(
            sim_matrix.round(3),
            index=[f"URL_{i}" for i in range(len(urls))],
            columns=[f"URL_{i}" for i in range(len(urls))],
        )
        st.dataframe(sim_df, use_container_width=True)

        st.subheader("Category Wise URLs")
        for category in sorted(result["Category"].unique()):
            with st.expander(
                f"{category} ({len(result[result['Category'] == category])} URLs)",
            ):
                subset = result[result["Category"] == category]
                for url in subset["URL"]:
                    st.write("•", url)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download clustered URLs as CSV",
            csv,
            "clustered_urls.csv",
            "text/csv",
        )