import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
import collections


# --- ⚙️ НАЛАШТУВАННЯ ---
FILE_IN = 'YOUR_FILE.csv'
FILE_OUT = 'duplicates_merged.csv'
# Налаштування колонок з вашого файла
#Назва колонки за якою визначаємо перший з дублікатів який було створенно
FIRST_RECORD = 'FIRST_RECORD'
#Назва колонки яка характерна для того щоб містити унікальну інформацію про об'єкт
COLUMN_ONE = 'COLUMN_ONE'
#Назва колонки яка характерна для того щоб містити унікальну інформацію про об'єкт
COLUMN_TWO = 'COLUMN_TWO'

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


# Поріг схожості для кластеризації. Можна налаштовувати, більше число - більш жорсткі вимоги до схожості
SIMILARITY_THRESHOLD = 0.5
BLOCKING_KEY = 'MAIN_COLOMN_NAME' # Колонка для попереднього групування


# --- 1. Завантаження моделі ---
def load_model_and_tokenizer(model_name):
    print(f"Завантаження моделі '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print("Модель успішно завантажено.")
        return tokenizer, model
    except Exception as e:
        print(f"Помилка при завантаженні моделі: {e}")
        exit()


def get_embeddings(texts, tokenizer, model, batch_size=16):
    model.eval()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = list(texts)[i:i+batch_size] # Переконуємось, що це список
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean Pooling - усереднюємо токени для отримання одного вектора
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings)


tokenizer, model = load_model_and_tokenizer(MODEL_NAME)


# --- 2. Економна загрузка і групування ---
print("Крок 1: Економне завантаження даних для пошуку дублікатів...")
# Завантажуємо ТІЛЬКИ необхідні для пошуку дублікатів колонки
try:
    use_cols = [{COLUMN_ONE}, {COLUMN_TWO}, BLOCKING_KEY, {FIRST_RECORD}]
    df_light = pd.read_csv(FILE_IN, sep=';', usecols=use_cols, low_memory=False)
    df_light[{FIRST_RECORD}] = pd.to_datetime(df_light[{FIRST_RECORD}], errors='coerce')
    print("Мінімальний набір даних завантажено.")
except Exception as e:
    print(f"Помилка при читанні файлу: {e}")
    exit()


# --- 3. Пошук дублікатів в групах ---
print("Крок 2: Пошук дублікатів всередині груп...")
duplicate_map = {} # Словник {id_дубліката: id_оригіналу}


# Групуємо DataFrame за ключем (наприклад, за власником)
grouped = df_light.groupby(BLOCKING_KEY)


for owner, group_df in tqdm(grouped, desc="Обробка груп"):
    if len(group_df) < 2:
        continue # Пропускаємо групи з одним записом


    names = group_df[{COLUMN_TWO}].fillna('missing').tolist()
    
    # Створюємо ембеддінги ТІЛЬКИ для поточної маленької групи
    embeddings = get_embeddings(names, tokenizer, model)


    # Кластеризація ТІЛЬКИ для поточної групи
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=SIMILARITY_THRESHOLD,
        metric='cosine',
        linkage='average'
    ).fit(embeddings.numpy())


    # Формуємо кластери
    clusters = collections.defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        clusters[label].append(i) # Зберігаємо індекси всередині групи


    for cluster_indices in clusters.values():
        if len(cluster_indices) < 2:
            continue


        # Знаходимо оригінал (найстаріший) всередині кластера
        cluster_df = group_df.iloc[cluster_indices]
        original = cluster_df.loc[cluster_df[{FIRST_RECORD}].idxmin()]
        
        # Записуємо дублікати у загальний словник
        for idx, row in cluster_df.iterrows():
            if row[{COLUMN_ONE}] != original[{COLUMN_ONE}]:
                duplicate_map[row[{COLUMN_ONE}]] = original[{COLUMN_ONE}]


print(f"Знайдено {len(duplicate_map)} дублікатів.")


# --- 4. Фінальна обробка з повним файлом ---
print("Крок 3: Завантаження повного файлу та об'єднання даних...")
df_full = pd.read_csv(FILE_IN, sep=';', low_memory=False)


# Створюємо колонку зі статусом дубліката
df_full['Статус дубліката'] = df_full[{COLUMN_ONE}].map(duplicate_map)


# Знаходимо оригінали та збагачуємо їх
original_ids = df_full['Статус дубліката'].dropna().unique()
for original_id in tqdm(original_ids, desc="Об'єднання даних в оригінали"):
    original_idx = df_full[df_full[{COLUMN_ONE}] == original_id].index
    if original_idx.empty: continue
    
    duplicates_df = df_full[df_full['Статус дубліката'] == original_id]
    
    for col in df_full.columns:
        if pd.isna(df_full.loc[original_idx[0], col]):
            first_valid_value = duplicates_df[col].dropna().values
            if len(first_valid_value) > 0:
                df_full.loc[original_idx[0], col] = first_valid_value[0]

# --- 5. Збереження результату ---
print("Сортування та збереження фінального файлу...")
if 'Статус дубліката' in df_full.columns:
    col_status = df_full.pop('Статус дубліката')
    df_full.insert(df_full.columns.get_loc(COLUMN_TWO) + 1, col_status.name, col_status)


df_full['is_duplicate'] = df_full[col_status.name].notna()
df_sorted = df_full.sort_values(by=['is_duplicate', col_status.name], ascending=[False, True])
df_sorted = df_sorted.drop(columns=['is_duplicate'])


df_sorted.to_csv(FILE_OUT, sep=';', index=False, encoding='utf-8-sig')
print(f"Готово! Результат збережено у файл: {FILE_OUT}")
