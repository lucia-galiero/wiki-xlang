import os
import re
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# Scarica il modello BERT LaBSE
tokenizer_labse = BertTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = BertModel.from_pretrained("sentence-transformers/LaBSE", device_map = 'cuda')

# Funzione per dividere i testi in frasi utilizzando il tokenizer di NLTK
def split_into_sentences(text):
    return sent_tokenize(text)

def get_labse_embedding(sentences):
    tokens = tokenizer_labse(sentences, return_tensors='pt', padding=True, truncation=True)
    tokens = {k: tokens[k].to('cuda') for k in tokens.keys()}
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Funzione per calcolare la cosine similarity
def calculate_cosine_similarity(embeddings1, embeddings2):
    similarity = cosine_similarity(embeddings1, embeddings2)
    return similarity

# Funzione per elaborare una sottocartella
def process_subdirectory(directory):
    files = os.listdir(directory)
    sentences_by_language = {}
    max_sentences = 0

    # Raggruppa i testi per lingua e memorizza il numero massimo di frasi
    for file in files:
        match = re.search(r'_([a-zA-Z]{2})\.txt$', file)
        if match:
            language = match.group(1)
            if language not in sentences_by_language:
                sentences_by_language[language] = []
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                text = f.read()
                sentences = split_into_sentences(text)
                sentences_by_language[language].extend(sentences)
                # Aggiorna il numero massimo di frasi
                max_sentences = max(max_sentences, len(sentences))

    # Calcola gli embeddings per ogni lingua
    embeddings_by_language = {}
    for language, sentences in sentences_by_language.items():
        sentence_embeddings = get_labse_embedding(sentences) # troviamo i tensori degli embedding delle frasi
        zero_tensor = torch.zeros(sentence_embeddings[0].shape[0]) # costruiamo un tensore di 0 con dimensione uguale a quella degli embeddings
        pad_length = (max_sentences - len(sentences)) # troviamo la lunghezza di padding
        pad_list = [zero_tensor] * pad_length # costruiamo la lista non-tensor di tensori
        if pad_list:
            pad_tensor = torch.vstack(pad_list).to('cuda') # se la lista ha elementi creiamo il tensore (vstack non funziona con liste vuote)
            embeddings_by_language[language] = torch.vstack([sentence_embeddings, pad_tensor]) # facciamo lo stack verticale (vstack appunto) degli embeddings e dei 0-tensors
        else:
            embeddings_by_language[language] = sentence_embeddings # se la lista è vuota mettiamo solo gli embedding nel dizionario 'embeddings_by_language'
        

    # Calcola la cosine similarity tra tutte le coppie di frasi
    similarities = []
    languages = list(embeddings_by_language.keys())
    for i in range(len(languages)):
        lang1 = languages[i]
        for j in range(i + 1, len(languages)):
            lang2 = languages[j]
            similarity = calculate_cosine_similarity(embeddings_by_language[lang1].cpu(), embeddings_by_language[lang2].cpu())
            for k in range(similarity.shape[0]):
                for l in range(similarity.shape[1]):
                    if k < len(sentences_by_language[lang1]) and l < len(sentences_by_language[lang2]):
                        similarities.append({
                            'Language 1': lang1,
                            'Sentence 1': sentences_by_language[lang1][k],
                            'Language 2': lang2,
                            'Sentence 2': sentences_by_language[lang2][l],
                            'Cosine similarity': similarity[k][l]
                        })

    # Crea un DataFrame con i risultati e restituiscilo
    df = pd.DataFrame(similarities)
    return df

# Funzione principale
def main():
    main_directory = "./data"
    subdirectories = os.listdir(main_directory)

    sim_dir = './similarities'

    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    progbar = tqdm(subdirectories, total = len(subdirectories))
    for subdirectory in progbar:
        progbar.set_description(desc=f'Current subdirectory: {subdirectory}') # scriviamo la lingua corrente nella progress bar
        subdirectory_path = os.path.join(main_directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            df = process_subdirectory(subdirectory_path)
            csv_filename = f"{subdirectory}.csv"
            df.to_csv(os.path.join(sim_dir, csv_filename), index=False)

if __name__ == "__main__":
    main()
