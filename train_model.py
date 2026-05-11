import pandas as pd
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
import pickle

df = pd.read_excel("deals.xlsx")

df = df.rename(columns={
    'cen_za_kv_m': 'price_per_sqm',
    'cena_zdelki': 'price_total',
    'Znachenie_osnovnoy_characteristici': 'area',
    'God_postroyki': 'build_year',
    'Mestopolozhenie': 'address',
    'Vid_sdelki': 'deal_type',
    'Naimenovanie': 'name',
    'Vid_obyekta_nedvizhimosti': 'object_type',
    'Vid_razreshennogo_ispolzovaniya': 'permitted_use',
    'Obyekty_t__nedvizhimosti': 'kadastr'
})

df = df[df['deal_type'] == 'Купля-продажа'].copy()
df = df[df['price_per_sqm'] > 100].copy()
df = df[df['area'] > 10].copy()
df['build_year'] = df['build_year'].fillna(2015)

type_map = {'Земельный участок': 1, 'Здание': 2, 'Помещение': 3, 'Сооружение': 4}
df['object_type_code'] = df['object_type'].map(type_map).fillna(0)

print(f"После фильтрации: {len(df)} сделок")

X = df[['area', 'build_year', 'object_type_code']].fillna(0)
y = df['price_per_sqm']

model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

df.to_csv("deals_clean.csv", index=False)
print("✅ Модель обучена!")
