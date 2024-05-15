import os
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
import glob
import re
import nltk
import random
from tqdm.auto import tqdm

main_directory = "./data"

subdirectories = [subdir for subdir in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, subdir))]

tokenizer_labse = BertTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = BertModel.from_pretrained("sentence-transformers/LaBSE")

# Funzione per ottenere l'embedding di una frase utilizzando LABSE
def get_labse_embedding(sentences):
    tokens = tokenizer_labse(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Itera su tutte le sottocartelle
for subdirectory in subdirectories:
    subdirectory_path = os.path.join(main_directory, subdirectory)
    # Trova tutti i file di testo nella sottocartella corrente
    file_paths = glob.glob(os.path.join(subdirectory_path, '*.txt'))

    # Verifica se la sottocartella corrente contiene file di testo
    if file_paths:
        # Esegue il codice per questa sottocartella

        languages = []
        data = {}

        # Carica i dati dai file di testo e suddividi le frasi
        for file_path in file_paths:
            lang = re.search(r'_([a-zA-Z]{2})\.txt$', file_path).group(1).upper()
            languages.append(lang)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                sentences = sent_tokenize(text)
                data[lang] = sentences

        # Trova la lunghezza massima tra tutti i dati
        max_length = max(len(data[lang]) for lang in languages)

        # Riempire o tagliare i dati in modo che abbiano la stessa lunghezza per tutte le lingue
        for lang in languages:
            while len(data[lang]) < max_length:
                # Se il numero di frasi è inferiore alla lunghezza massima, aggiungi casualmente una frase dai dati esistenti
                data[lang].append(random.choice(data[lang]))
            if len(data[lang]) > max_length:
                # Se il numero di frasi è superiore alla lunghezza massima, taglia i dati
                data[lang] = data[lang][:max_length]

        # Calcola l'embedding per le frasi nelle diverse lingue
        embeddings = {}
        for lang in tqdm(languages, total = len(languages), desc=f'{subdirectory}'):
            data_lang = data[lang]
            embeddings[lang] = get_labse_embedding(data_lang)

        # Calcola la similarità coseno tra i vettori di embedding per le diverse coppie di lingue
        cosine_similarities = {}
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i < j:  # Per evitare duplicati, calcoliamo solo una volta per ciascuna coppia (lang1, lang2) dove lang1 < lang2
                    key = f"Cosine_{lang1}_{lang2}"
                    cosine_similarities[key] = [cosine_similarity([emb1], [emb2])[0][0] for emb1, emb2 in zip(embeddings[lang1], embeddings[lang2])]

        # Costruisci un DataFrame per i risultati
        df_pairs = pd.DataFrame(columns=["LANGUAGE1", "SENTENCE1", "LANGUAGE2", "SENTENCE2", "COSINE SIMILARITY"])

        # Iterate over the languages to extract unique sentence pairs and cosine similarities
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i < j:  # Per evitare duplicati, consideriamo solo una volta per ciascuna coppia (lang1, lang2) dove lang1 < lang2
                    key = f"Cosine_{lang1}_{lang2}"
                    if key in cosine_similarities:
                        # Extract unique sentence pairs and cosine similarities for the language pair
                        sentences1 = data[lang1]
                        sentences2 = data[lang2]
                        similarities = cosine_similarities[key]

                        # Create a DataFrame for the language pair
                        df_lang_pair = pd.DataFrame({
                            "LANGUAGE1": [lang1] * len(sentences1),
                            "SENTENCE1": sentences1,
                            "LANGUAGE2": [lang2] * len(sentences2),
                            "SENTENCE2": sentences2,
                            "COSINE SIMILARITY": similarities
                        })

                        # Append the DataFrame to df_pairs
                        df_pairs = pd.concat([df_pairs, df_lang_pair], ignore_index=True)

        # Write the DataFrame to an Excel file
        output_csv_path = os.path.join('./similarities', f"{subdirectory}_similarity.csv")
        df_pairs.to_csv(output_csv_path, index=False)