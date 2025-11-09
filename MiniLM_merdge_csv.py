import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
import collections

# --- ⚙️ НАСТРОЙКИ ---
FILE_IN = 'restorants.csv'
FILE_OUT = 'restorants_final_merged_optimized.csv'

# VVV ИСПРАВЛЕНИЕ: Используем правильную и эффективную модель VVV
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Порог схожести для кластеризации. Можете подбирать.
SIMILARITY_THRESHOLD = 0.5
BLOCKING_KEY = 'Organization - Owner' # Колонка для предварительной группировки

# --- 1. Загрузка модели ---
def load_model_and_tokenizer(model_name):
    print(f"Загрузка модели '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print("Модель успешно загружена.")
        return tokenizer, model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        exit()

def get_embeddings(texts, tokenizer, model, batch_size=16):
    model.eval()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = list(texts)[i:i+batch_size] # Убедимся, что это список
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean Pooling - усредняем токены для получения одного вектора
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings)

tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

# --- 2. Экономная загрузка и группировка ---
print("Шаг 1: Экономная загрузка данных для поиска дублей...")
# Загружаем ТОЛЬКО необходимые для поиска дублей колонки
try:
    use_cols = ['Organization - ID', 'Organization - Name', BLOCKING_KEY, 'Organization - Organization created']
    df_light = pd.read_csv(FILE_IN, sep=';', usecols=use_cols, low_memory=False)
    df_light['Organization - Organization created'] = pd.to_datetime(df_light['Organization - Organization created'], errors='coerce')
    print("Минимальный набор данных загружен.")
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    exit()

# --- 3. Поиск дублей по группам ---
print("Шаг 2: Поиск дубликатов внутри групп...")
duplicate_map = {} # Словарь {id_дубликата: id_оригинала}

# Группируем DataFrame по ключу (например, по владельцу)
grouped = df_light.groupby(BLOCKING_KEY)

for owner, group_df in tqdm(grouped, desc="Обработка групп"):
    if len(group_df) < 2:
        continue # Пропускаем группы с одной записью

    names = group_df['Organization - Name'].fillna('missing').tolist()
    
    # Создаем эмбеддинги ТОЛЬКО для текущей маленькой группы
    embeddings = get_embeddings(names, tokenizer, model)

    # Кластеризация ТОЛЬКО для текущей группы
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=SIMILARITY_THRESHOLD,
        metric='cosine',
        linkage='average'
    ).fit(embeddings.numpy())

    # Собираем кластеры
    clusters = collections.defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        clusters[label].append(i) # Сохраняем индексы внутри группы

    for cluster_indices in clusters.values():
        if len(cluster_indices) < 2:
            continue

        # Находим оригинал (самый старый) внутри кластера
        cluster_df = group_df.iloc[cluster_indices]
        original = cluster_df.loc[cluster_df['Organization - Organization created'].idxmin()]
        
        # Записываем дубликаты в общий словарь
        for idx, row in cluster_df.iterrows():
            if row['Organization - ID'] != original['Organization - ID']:
                duplicate_map[row['Organization - ID']] = original['Organization - ID']

print(f"Найдено {len(duplicate_map)} дубликатов.")

# --- 4. Финальная обработка с полным файлом ---
print("Шаг 3: Загрузка полного файла и слияние данных...")
df_full = pd.read_csv(FILE_IN, sep=';', low_memory=False)

# Создаем колонку со статусом дубликата
df_full['Duplicate status'] = df_full['Organization - ID'].map(duplicate_map)

# Находим оригиналы и обогащаем их
original_ids = df_full['Duplicate status'].dropna().unique()
for original_id in tqdm(original_ids, desc="Слияние данных в оригиналы"):
    original_idx = df_full[df_full['Organization - ID'] == original_id].index
    if original_idx.empty: continue
    
    duplicates_df = df_full[df_full['Duplicate status'] == original_id]
    
    for col in df_full.columns:
        if pd.isna(df_full.loc[original_idx[0], col]):
            first_valid_value = duplicates_df[col].dropna().values
            if len(first_valid_value) > 0:
                df_full.loc[original_idx[0], col] = first_valid_value[0]

# Помечаем дубликаты на удаление
dup_indices = df_full[df_full['Duplicate status'].notna()].index
if 'Organization - Website' in df_full.columns:
    df_full.loc[dup_indices, 'Organization - Website'] = 'delete'
    
# --- 5. Сохранение результата ---
print("Сортировка и сохранение финального файла...")
if 'Duplicate status' in df_full.columns:
    col_status = df_full.pop('Duplicate status')
    df_full.insert(df_full.columns.get_loc('Organization - Name') + 1, col_status.name, col_status)

df_full['is_duplicate'] = df_full[col_status.name].notna()
df_sorted = df_full.sort_values(by=['is_duplicate', col_status.name], ascending=[False, True])
df_sorted = df_sorted.drop(columns=['is_duplicate'])

df_sorted.to_csv(FILE_OUT, sep=';', index=False, encoding='utf-8-sig')
print(f"Готово! Результат сохранен в файл: {FILE_OUT}")