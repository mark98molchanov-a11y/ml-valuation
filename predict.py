import pickle
import sys
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def is_empty(val):
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in ['', 'nan', 'none', 'null', '-', 'нет']

def clean_val(val, max_len=80):
    if pd.isna(val) or str(val).strip().lower() in ['nan', 'none', '']:
        return ''
    return str(val).strip()[:max_len]

# Загружаем модели
try:
    with open("model_buildings.pkl", "rb") as f:
        model_buildings = pickle.load(f)
except:
    model_buildings = None

try:
    with open("model_land.pkl", "rb") as f:
        model_land = pickle.load(f)
except:
    model_land = None

df = pd.read_csv("deals_clean.csv")

# Параметры
area = float(sys.argv[1])
build_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
object_type = sys.argv[3] if len(sys.argv) > 3 else 'Помещение'
permitted_use = sys.argv[4] if len(sys.argv) > 4 else ''
address = sys.argv[5] if len(sys.argv) > 5 else ''
kadastr = sys.argv[6] if len(sys.argv) > 6 else ''
wall_material = sys.argv[7] if len(sys.argv) > 7 else ''
object_name = sys.argv[8] if len(sys.argv) > 8 else ''

type_map = {'Земельный участок': 1, 'Здание': 2, 'Помещение': 3, 'Сооружение': 4}
type_code = type_map.get(object_type, 0)
is_land = (type_code == 1)

material_map = {'Кирпич': 1, 'Панель': 2, 'Монолит': 3, 'Дерево': 4, 'Блок': 5}
wall_code = material_map.get(wall_material, 0)

# Город
city = ''
if address:
    all_cities = set()
    for addr in df['address'].dropna():
        parts = addr.replace(',', ' ').replace('.', ' ').split()
        for word in parts:
            if word[0].isupper() and len(word) > 3:
                all_cities.add(word)
    
    address_parts = address.replace(',', ' ').split()
    for part in address_parts:
        if part in all_cities:
            city = part
            break
    
    if not city:
        known_cities = ['Салехард', 'Новый', 'Ноябрьск', 'Тарко-Сале', 'Надым',
                        'Губкинский', 'Муравленко', 'Лабытнанги', 'Красноселькуп']
        for c in known_cities:
            if c in address:
                city = c
                break

# ============================================================
# ML-прогноз
# ============================================================
if is_land and model_land:
    use_code = pd.factorize(df['permitted_use'])[0][df['permitted_use'] == permitted_use]
    use_code = use_code[0] if len(use_code) > 0 else 0
    price_sqm = model_land.predict([[area, build_year, use_code]])[0]
elif not is_land and model_buildings:
    price_sqm = model_buildings.predict([[area, build_year, type_code, wall_code]])[0]
else:
    price_sqm = df['price_per_sqm'].median()

price_total = price_sqm * area

# ============================================================
# Подбор аналогов
# ============================================================
similar = df[df['object_type_code'] == type_code].copy()
similar = similar[similar['area'].between(area * 0.3, area * 3.0)]

# Для земли — фильтр по permitted_use (ВРИ)
if is_land and permitted_use:
    similar = similar[~similar['permitted_use'].apply(is_empty)].copy()
    use_filtered = similar[similar['permitted_use'].str.contains(permitted_use[:20], na=False)]
    if len(use_filtered) >= 3:
        similar = use_filtered

# Для зданий/помещений — фильтр по наименованию и материалу стен
if not is_land:
    if object_name:
        name_filtered = similar[similar['name'].astype(str).str.contains(object_name[:20], na=False)]
        if len(name_filtered) >= 3:
            similar = name_filtered
    
    if wall_material:
        material_filtered = similar[similar['wall_material'].astype(str).str.contains(wall_material, na=False)]
        if len(material_filtered) >= 3:
            similar = material_filtered

# Город
if city and len(similar) >= 5:
    city_filtered = similar[similar['address'].str.contains(city, na=False)]
    if len(city_filtered) >= 3:
        similar = city_filtered

# Расширяем если пусто
if len(similar) < 5:
    similar = df[df['object_type_code'] == type_code].copy()
if len(similar) < 3:
    similar = df.copy()

n_neighbors = min(5, len(similar))
nn = NearestNeighbors(n_neighbors=n_neighbors)

if is_land:
    nn.fit(similar[['area', 'build_year']].fillna(0).values)
else:
    nn.fit(similar[['area', 'build_year', 'object_type_code']].fillna(0).values)

distances, indices = nn.kneighbors([[area, build_year] + ([type_code] if not is_land else [])])
analogs = similar.iloc[indices[0]]

# Корректировки
corrections = []
for _, analog in analogs.iterrows():
    corr = 1.0
    if area > 100 and analog['area'] < 100:
        corr *= 0.95
    elif area < 50 and analog['area'] > 50:
        corr *= 1.05
    year_diff = build_year - analog['build_year']
    if abs(year_diff) > 5:
        corr *= 1 + (year_diff * 0.005)
    if city and city not in str(analog.get('address', '')):
        corr *= 0.90
    corrections.append(round(corr, 3))

# Средневзвешенная
total_weight = 0
weighted_sum = 0
for i, (_, analog) in enumerate(analogs.iterrows()):
    weight = 1 / (distances[0][i] + 0.01)
    adjusted_price = analog['price_per_sqm'] * corrections[i]
    weighted_sum += adjusted_price * weight
    total_weight += weight
weighted_avg_price = weighted_sum / total_weight if total_weight > 0 else price_sqm

# Обоснование
avg_analog = analogs['price_per_sqm'].mean()
diff_pct = (price_sqm - avg_analog) / avg_analog * 100

justification = f"""ОЦЕНКА ОБЪЕКТА{' с КН ' + kadastr if kadastr else ''}:
Тип: {object_type} | Площадь: {area:.0f} м² | Год: {build_year} | Город: {city if city else 'не определён'}"""

if is_land and permitted_use:
    justification += f"\nВРИ: {permitted_use}"
if not is_land:
    if object_name:
        justification += f"\nНаименование: {object_name}"
    if wall_material:
        justification += f"\nМатериал стен: {wall_material}"

justification += f"""

ЭТАП 1: ПРЕДВАРИТЕЛЬНЫЙ ОТБОР
Из базы {len(df)} сделок отобраны по критериям: тип={object_type}, площадь={area*0.3:.0f}-{area*3.0:.0f} м²"""
if city:
    justification += f", город={city}"
if is_land and permitted_use:
    justification += f", ВРИ={permitted_use}"
if not is_land and object_name:
    justification += f", наименование={object_name}"
if not is_land and wall_material:
    justification += f", материал={wall_material}"
justification += f"\nОтобрано: {len(similar)} объектов\n\nЭТАП 2: ФИНАЛЬНЫЙ ОТБОР 5 АНАЛОГОВ\n"

for i, (_, a) in enumerate(analogs.iterrows(), 1):
    kad = clean_val(a.get('kadastr', ''), 20)
    name = clean_val(a.get('name', ''), 50)
    justification += f"Аналог {i}: {name} | КН: {kad} | Площадь: {int(a['area'])} м² | Цена: {int(a['price_per_sqm'])} руб/м² | Корр: {corrections[i-1]:.3f}\n"

justification += f"""
ЭТАП 3: РАСЧЁТ
ML-прогноз: {price_sqm:.0f} руб/м² | Среднее аналогов: {avg_analog:.0f} руб/м² | Средневзвешенное: {weighted_avg_price:.0f} руб/м²
Финальная цена за м²: {price_sqm:.0f} руб/м² | Общая стоимость: {price_total:.0f} руб.

ЭТАП 4: ЗАКЛЮЧЕНИЕ
Рыночная стоимость определена в размере {price_total:.0f} руб. ({price_sqm:.0f} руб/м²). Отклонение от среднего аналогов: {diff_pct:+.1f}%."""

# Итоговый JSON
result = {
    "object": {
        "kadastr": clean_val(kadastr, 20),
        "area": area,
        "build_year": build_year,
        "object_type": object_type,
        "permitted_use": clean_val(permitted_use, 50),
        "name": clean_val(object_name, 80),
        "wall_material": clean_val(wall_material, 30),
        "city": clean_val(city, 30),
        "address": clean_val(address, 120)
    },
    "predicted": {
        "price_per_sqm": round(price_sqm),
        "price_total": round(price_total)
    },
    "calculation": {
        "ml_prediction": round(price_sqm),
        "avg_analogs": round(avg_analog),
        "weighted_avg": round(weighted_avg_price),
        "deviation_pct": round(diff_pct, 1)
    },
    "justification": justification,
    "analogs": []
}

for i, (_, a) in enumerate(analogs.iterrows()):
    result["analogs"].append({
        "num": i + 1,
        "kadastr": clean_val(a.get('kadastr', ''), 20),
        "name": clean_val(a.get('name', ''), 80),
        "area": round(float(a['area']), 1),
        "price_per_sqm": round(float(a['price_per_sqm'])),
        "price_total": round(float(a.get('price_total', 0))),
        "build_year": int(a.get('build_year', 0)) if not pd.isna(a.get('build_year')) else 0,
        "object_type": clean_val(a.get('object_type', ''), 30),
        "permitted_use": clean_val(a.get('permitted_use', ''), 50),
        "wall_material": clean_val(a.get('wall_material', ''), 30),
        "address": clean_val(a.get('address', ''), 120),
        "correction": corrections[i],
        "similarity": round(100 - distances[0][i] * 15, 1)
    })

print(json.dumps(result, ensure_ascii=False, indent=2))
