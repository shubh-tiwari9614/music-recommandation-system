# Music Recommendation System: A Content-Based Filtering Approach

An intelligent system that leverages Natural Language Processing (NLP) and Machine Learning to recommend songs based on a user's selected track. It analyzes metadata like genre, artist, and track name to find and suggest the most similar songs from a large dataset.

This project was developed as a demonstration of content-based filtering, utilizing TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to perform real-time, context-aware song recommendations.

## Core AI/ML Concepts

This system is architected to explicitly demonstrate foundational theories in Machine Learning and Natural Language Processing:

- **Content-Based Filtering:** Unlike collaborative filtering, this system relies solely on the properties of the items (songs) themselves. It recommends songs that are inherently similar to the one the user likes, without needing data from other users.
- **Natural Language Processing (NLP):** The system applies NLP techniques to convert textual song metadata into a format that a machine can understand.
- **TF-IDF Vectorization:** This technique transforms the combined textual data (genre, artist, track name) into a numerical matrix, weighting words based on their frequency in a specific song against their frequency across all songs. This helps in capturing the unique "character" of each track.
- **Cosine Similarity:** A mathematical metric used to calculate the cosine of the angle between two non-zero vectors. In this context, it measures the similarity between two songs by comparing their TF-IDF vectors. A higher cosine similarity score indicates a higher degree of similarity.

## System Architecture: How It Works

The project follows a modular, script-based architecture for clarity and ease of use:

1.  **Data Acquisition:** The system loads a comprehensive music dataset (`tcc_ceds_music.csv`) which contains information on thousands of songs, including their `track_name`, `artist_name`, and `genre`.
2.  **Preprocessing & Feature Engineering:** A new feature, `combined_features`, is created by concatenating the `genre`, `artist_name`, and `track_name` of each song. This rich text string serves as the primary data source for similarity calculations.
3.  **Vectorization:** The `combined_features` are processed using `TfidfVectorizer`, converting them into a sparse matrix (`tfidf_matrix`). Each row in this matrix is a unique vector representation of a song.
4.  **Similarity Computation:** A pairwise `cosine_similarity` matrix is calculated. This matrix holds the similarity score for every song against every other song in the dataset.
5.  **Recommendation Engine:** When a user inputs a song title, the system:
    - Finds the index of that song in the dataset.
    - Retrieves its corresponding row from the cosine similarity matrix.
    - Sorts all other songs by their similarity score in descending order.
    - Returns the top `N` most similar tracks as the recommendation.
6.  **Data Visualization:** The project includes exploratory data analysis (EDA) and result visualization using `matplotlib` and `seaborn`, providing insights into genre distributions and a clear display of the recommendations.

## Setup & Installation

### Prerequisites:

- Python 3.8+
- pip (Python package manager)

### Installation

1.  **Clone the Repository (or download the notebook)**

    ```bash
    git clone https://github.com/your-username/music-recommendation-system.git
    cd music-recommendation-system
    ```
2.  **Install Required Libraries**

    ```bash
pip install -r requirements.txt
    ```
3.  **Download the Dataset**
The system uses the tcc_ceds_music.csv dataset. When you run the notebook, it will prompt you to upload this file



## Execution & Usage
The system is designed to be run as a Jupyter Notebook (Musicrecommendationsystem.ipynb). Follow the steps within the notebook for a guided walkthrough.

**Scenario 1**: Exploratory Data Analysis (EDA)
The notebook begins by performing EDA to understand the dataset. This includes generating visualizations for:

-Top 10 Genres: A bar chart showing the most frequent genres in the dataset.

-Top 10 Artists by Song Count: A bar chart identifying the most prolific artists.

**Scenario 2**: Generating a Recommendation
Upload Data: Execute the cell that prompts you to upload the tcc_ceds_music.csv file.

Enter a Song: Modify the get_recommendations function call with your desired song title.

```bash
recommended_songs = get_recommendations('hum tujhse mohabbat kar ke', data, cosine_sim, top_n=10)
View Results: The notebook will output a table of the top 10 recommended songs, including their track_name, artist_name, and genre. It will also generate a bar chart visualizing the recommendations.
``` 

Example Output:

```bash
[>] Calculating recommendations for 'hum tujhse mohabbat kar ke'...
[+] RECOMMENDATIONS SECURED!

                                track_name     artist_name    genre
0                     mohabbat bhi jhoothi          mukesh      pop
33                raat andheri door savera          mukesh      pop
27825                       ini illaye hum  hiphop tamizha  hip hop
1946                 zindagi ke safar mein   kishore kumar      pop
27500  ya ke'sto no es pop, es hip-hop/rap         frank t  hip hop
...and more.
```

Repository Map

```bash
Musicrecommendationsystem.ipynb - The main Jupyter Notebook containing all code for EDA, model building, and execution.

tcc_ceds_music.csv - The dataset containing music metadata (to be uploaded).

requirements.txt - A file listing all Python dependencies (e.g., pandas, scikit-learn, matplotlib, seaborn).

README.md - This file.
```
#Author
##Made by: Shubh Tiwari 
25BAI10799



