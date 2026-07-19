def apply_nlp(
        dataset, 
        column_name
    ):

    import numpy as np
    import pandas as pd
    from IPython.display import display
    from pathlib import Path
    import matplotlib.pyplot as plt
    import re

    from collections import Counter
    from sklearn.feature_extraction.text import CountVectorizer
    from deep_translator import GoogleTranslator

    import spacy
    translator = GoogleTranslator(source="pt", target="en")

    #Establish parameters for NLP analysis

    nlp = spacy.load(
        "pt_core_news_sm",
        disable=["ner", "parser"]
    )

    vectorizer = CountVectorizer(
        ngram_range=(3,3),
        min_df=5
    )

    custom_stopwords = {
        "produto"
    }

    #Main part
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-záàâãéêíóôõúç\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        doc = nlp(text)

        words = []

        for token in doc:
            if token.is_alpha and not token.is_stop and token.lemma_ not in custom_stopwords:
                words.append(token.lemma_)
        
        words = " ".join(words)
        return words
        
    dataset = dataset.copy()
    dataset = dataset.dropna(subset=[column_name])

    dataset["processed_text"] = dataset[column_name].apply(clean_text)
    all_words = " ".join(dataset["processed_text"]).split()

    word_counts = Counter(all_words)

    top_word_counts_pd = pd.DataFrame(
        word_counts.most_common(30),
        columns=["word", "count"]
    )

    top_word_counts_pd["translated_words"] = top_word_counts_pd["word"].apply(translator.translate)

    matrix = vectorizer.fit_transform(
        dataset["processed_text"]
    )

    counts = matrix.sum(axis=0).A1
    bigrams = vectorizer.get_feature_names_out()


    bigram_df = pd.DataFrame({
        "bigram": bigrams,
        "count": counts  
    })
    bigram_df = bigram_df.sort_values("count", ascending=False).head(30)

    bigram_df["translated_bigrams"] = bigram_df["bigram"].apply(translator.translate)

    return bigram_df

    


