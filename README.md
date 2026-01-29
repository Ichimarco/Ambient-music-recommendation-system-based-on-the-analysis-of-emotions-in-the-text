# AI Music Emotion Recommender

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![NLP](https://img.shields.io/badge/AI-HuggingFace-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

This research application was developed as part of a diploma thesis. The system utilizes Natural Language Processing (NLP) and Fuzzy Logic to recommend musical tracks based on emotions detected in literary texts.

## Project Objective

The primary objective of this project is to evaluate the effectiveness of AI algorithms in selecting background music for reading. The application conducts a "Blind Test," where users rate the relevance of a musical track to a text passage without knowing whether the track was selected by the algorithm or assigned randomly.

The study uses excerpts from **"The Adventures of Sherlock Holmes"** by Arthur Conan Doyle as the testing material.

## Methodology

The system operates using a four-stage decision pipeline:

1.  **Text Analysis (NLP)**
    * Model: `Panda0116/emotion-classification-model` (Transformer-based).
    * Function: Classifies the input text into one of five emotion categories: *Happiness, Sadness, Anger, Fear, Tenderness*.

2.  **Fuzzy Logic Inference**
    * Library: `scikit-fuzzy`.
    * Function: Maps the detected emotion category and the model's confidence score onto a continuous 2D psychological space: **Valence** (Positivity) and **Arousal** (Energy).

3.  **Rule-Based Filtering**
    * Logic: Applies music theory rules to filter genres and musical keys.
    * Example: If *Anger* is detected (High Energy, Negative Valence), the system enforces aggressive genres (e.g., Industrial, Metal) and a Minor Key.

4.  **Recommendation Engine**
    * Function: Searches a dataset of approximately 114,000 Spotify tracks to find the nearest neighbor to the calculated Valence and Energy parameters.

## User Instructions

The application is deployed and available at: **[INSERT_YOUR_STREAMLIT_LINK_HERE]**

1.  Click the **"Randomize Text & Music (START)"** button.
2.  The system will generate a random text excerpt and select a music track.
3.  Read the text and listen to the Spotify preview.
4.  Rate the compatibility of the music with the text on a scale of 1 to 10.
5.  Repeat the test at least **5 times**.
6.  Navigate to the sidebar, click **"DOWNLOAD RESULTS (CSV)"**, and send the file to the author.

## Local Installation

To run this project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/music-emotion-thesis.git](https://github.com/YourUsername/music-emotion-thesis.git)
    cd music-emotion-thesis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app_experiment.py
    ```

## File Structure

* `app_experiment.py` - The main application script containing the UI, Logic, and Models.
* `dataset.csv` - The database of Spotify tracks.
* `requirements.txt` - List of Python dependencies.
* `README.md` - Project documentation.

## Data Sources

* **Music Data:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset) by Maharshi Pandya.
* **Literary Texts:** Project Gutenberg (*The Adventures of Sherlock Holmes*).

## Author

This project was created for academic research purposes.