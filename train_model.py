import pandas as pd
from catboost import CatBoostRegressor
import pickle

def is_empty(val):
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in ['', 'nan', 'none', 'null', '-', 'нет']

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
    'Obyekty_t__nedvizhimosti': 'kadastr',
    'Material_sten': 'wall_material'  # ← добавили
})

df = df[df['deal_type'] == 'Купля-продажа'].copy()
df = df[df['price_per_sqm'] > 100].copy()
df = df[df['area'] > 10].copy()
df['build_year'] = df['build_year'].fillna(2015)

# Кодируем тип объекта
type_map = {'Земельный участок': 1, 'Здание': 2, 'Помещение': 3, 'Сооружение': 4}
df['object_type_code'] = df['object_type'].map(type_map).fillna(0)

# Кодируем материал стен
material_map = {'Кирпич': 1, 'Панель': 2, 'Монолит': 3, 'Дерево': 4, 'Блок': 5}
df['wall_material_code'] = df['wall_material'].map(material_map).fillna(0)

# ============================================================
# Разделяем на две группы для обучения
# ============================================================

# Группа 1: Здания + Помещения + Сооружения (учитываем name + wall_material)
buildings = df[df['object_type_code'].isin([2, 3, 4])].copy()
if len(buildings) > 10:
    X_buildings = buildings[['area', 'build_year', 'object_type_code', 'wall_material_code']].fillna(0)
    y_buildings = buildings['price_per_sqm']
    
    model_buildings = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0)
    model_buildings.fit(X_buildings, y_buildings)
    with open("model_buildings.pkl", "wb") as f:
        pickle.dump(model_buildings, f)
    print(f"✅ Модель для зданий/помещений обучена на {len(buildings)} объектах")

# Группа 2: Земельные участки (учитываем permitted_use)
land = df[df['object_type_code'] == 1].copy()
land = land[~land['permitted_use'].apply(is_empty)].copy()

if len(land) > 10:
    # Кодируем permitted_use в числа
    land['use_code'] = pd.factorize(land['permitted_use'])[0]
    
    X_land = land[['area', 'build_year', 'use_code']].fillna(0)
    y_land = land['price_per_sqm']
    
    model_land = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0)
    model_land.fit(X_land, y_land)
    with open("model_land.pkl", "wb") as f:
        pickle.dump(model_land, f)
    print(f"✅ Модель для земельных участков обучена на {len(land)} объектах")

# Сохраняем данные
df.to_csv("deals_clean.csv", index=False)
print("✅ Данные сохранены!")
