import pandas as pd
from catboost import CatBoostRegressor
import pickle

# Загружаем базу сделок
df = pd.read_excel("deals.xlsx")

# Переименовываем нужные колонки
df = df.rename(columns={
    'cen_za_kv_m': 'price_per_sqm',           # Цена за м²
    'cena_zdelki': 'price_total',              # Общая цена
    'Znachenie_osnovnoy_characteristici': 'area',  # Площадь
    'God_postroyki': 'build_year',             # Год постройки
    'Mestopolozhenie': 'address',              # Адрес
    'Vid_sdelki': 'deal_type',                 # Тип сделки
    'Naimenovanie': 'name'                     # Наименование
})

# Фильтруем только "Купля-продажа"
df = df[df['deal_type'] == 'Купля-продажа'].copy()

# Убираем строки без цены или без площади
df = df[df['price_per_sqm'] > 100].copy()
df = df[df['area'] > 10].copy()

# Заполняем пропуски
df['build_year'] = df['build_year'].fillna(2015)

print(f"После фильтрации: {len(df)} сделок")

# Обучаем модель на трёх признаках
X = df[['area', 'build_year']].fillna(0)
y = df['price_per_sqm']

model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0)
model.fit(X, y)

# Сохраняем модель
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Модель обучена и сохранена!")
