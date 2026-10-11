# UGC Cinemas – Customer Review Analysis (NLP)

NLP project (M2 Data Science, Dauphine-PSL). I collected Google Maps reviews of UGC cinemas in France and used them to understand what drives customer satisfaction and to predict a review's rating from its text alone.

The project has three parts:

1. **Exploratory data analysis**: distribution of ratings, time effects, user profiles and differences between cinemas.
2. **Topic modeling**: which themes (cleanliness, staff, price, comfort...) come up in the reviews (LDA and BERTopic).
3. **Text classification**: predict the rating (5 classes, then 3 classes) from the review text (TF-IDF + SVM / XGBoost, and a fine-tuned CamemBERT).

## Dataset

- **Source**: Google Maps reviews of UGC cinemas, collected with the [Serper](https://serper.dev) API (`recup_avis_ugc.ipynb`), newest first, in French and English.
- **Size**: 203,186 raw reviews. About 57% have no text and were dropped, leaving **86,720 reviews with text** from 37 cinemas. Language detection (`langdetect`) finds about 86% French.
- **Fields**: rating, date, text, likes, reviewer name, reviewer's total number of reviews and photos, cinema.

## Repository

| File | Content |
|---|---|
| `recup_avis_ugc.ipynb` | Review collection through the Serper API (pagination per cinema and language) |
| `Exploration_modelisation.ipynb` | Cleaning, EDA, topic modeling and classification |
| `Data/avis_ugc.csv` | Collected reviews |

## Method

**Preprocessing**: date conversion, removal of reviews without text, language detection, lemmatization with `simplemma`, French stopwords (NLTK) completed with cinema-specific words ("film", "ugc", "cinéma"...). Topic models only use French reviews of more than 3 words.

**Topic modeling**
- **LDA** on bigrams (63,871 reviews, 11,690 terms), K = 10 chosen from the log-likelihood curve, explored with pyLDAvis.
- **BERTopic** on raw text: multilingual MiniLM sentence embeddings, UMAP, HDBSCAN, c-TF-IDF, reduced from 693 clusters to 10 topics.

**Classification**
- TF-IDF (unigrams + bigrams) with a linear SVM (balanced class weights) and with XGBoost.
- `camembert-base` fine-tuned with Hugging Face (3 epochs, max length 128, fp16), about 30 to 35 minutes per run on an RTX 4070 laptop GPU.
- 80/20 train/test split (17,344 test reviews). 3-class version: negative (1-2 stars), neutral (3), positive (4-5).

## Results

**EDA**
- Average rating 3.78/5. Strongly skewed towards positive reviews (about 35,000 five-star and 25,000 four-star reviews), so class imbalance has to be handled.
- Negative reviews are longer: unhappy customers explain what went wrong, satisfied ones write a few words.
- Occasional reviewers give lower ratings (3.04 on average) than frequent contributors (3.95).
- Average rating peaked around 2019-2021 and has slightly decreased since. Ratings are stable across months, with a review peak in August.
- Cinemas differ clearly: the best ones average above 4, the weakest around 3.

**Topics**
- **Pain points**: price, especially concessions (popcorn), cleanliness of rooms and toilets, and some staff behaviour.
- **Strengths**: seat comfort, large screens and spacious rooms, welcoming and friendly staff.
- BERTopic also isolates smaller themes (sound volume, accessibility, Wi-Fi) that LDA does not separate.

**Rating prediction** (accuracy / macro F1 on the test set)

| Model | 5 classes | 3 classes |
|---|---|---|
| TF-IDF + Linear SVM | **0.80 / 0.81** | **0.92 / 0.88** |
| TF-IDF + XGBoost | 0.59 / 0.52 | 0.80 / 0.64 |
| CamemBERT (fine-tuned) | 0.69 / 0.66 | 0.88 / 0.79 |

- The linear SVM is the best model in both settings. XGBoost struggles on sparse TF-IDF features.
- With 3 classes, the SVM reaches F1 of 0.90 (negative), 0.77 (neutral) and 0.96 (positive). The neutral class is the hardest, since those reviews mix praise and complaints.
- With 5 classes, most errors are between adjacent ratings (for example 4 vs 5 stars).
- CamemBERT stays below the SVM and looks under-trained after 3 epochs. Training longer is the first improvement to try.

## Limitations and next steps

- Google Maps reviews are self-selected, and the 57% of ratings without text are not analysed.
- Imbalanced classes, and results driven by the largest cinemas.
- Next steps: more training epochs or a larger model, cinema-level topic analysis to explain the gaps between locations, and a time analysis of topics.
