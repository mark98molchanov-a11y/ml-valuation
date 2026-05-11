import pickle
import sys
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("deals_clean.csv")

# Параметры
area = float(sys.argv[1])
build_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
object_type = sys.argv[3] if len(sys.argv) > 3 else 'Помещение'
permitted_use = sys.argv[4] if len(sys.argv) > 4 else ''
address = sys.argv[5] if len(sys.argv) > 5 else ''
kadastr = sys.argv[6] if len(sys.argv) > 6 else ''

type_map = {'Земельный участок': 1, 'Здание': 2, 'Помещение': 3, 'Сооружение': 4}
type_code = type_map.get(object_type, 0)

# ============================================================
# Автоматическое определение города
# ============================================================
city = ''
if address:
    # Собираем все уникальные города из базы сделок
    all_cities = set()
    for addr in df['address'].dropna():
        # Извлекаем слова, которые могут быть городами
        parts = addr.replace(',', ' ').replace('.', ' ').split()
        for i, word in enumerate(parts):
            if word[0].isupper() and len(word) > 3:
                all_cities.add(word)
    
    # Ищем совпадения в адресе объекта
    address_parts = address.replace(',', ' ').split()
    for part in address_parts:
        if part in all_cities:
            city = part
            break
    
    # Если не нашли — ищем через список крупных городов
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
price_sqm = model.predict([[area, build_year, type_code]])[0]
price_total = price_sqm * area

# ============================================================
# Подбор аналогов
# ============================================================
similar = df[df['object_type_code'] == type_code].copy()
similar = similar[similar['area'].between(area * 0.3, area * 3.0)]

# Фильтр по городу
if city:
    city_filtered = similar[similar['address'].str.contains(city, na=False)]
    if len(city_filtered) >= 3:
        similar = city_filtered

# Фильтр по виду разрешенного использования
if permitted_use:
    use_filtered = similar[similar['permitted_use'].astype(str).str.contains(permitted_use[:20], na=False)]
    if len(use_filtered) >= 3:
        similar = use_filtered

if len(similar) < 5:
    similar = df[df['object_type_code'] == type_code].copy()

nn = NearestNeighbors(n_neighbors=min(5, len(similar)))
nn.fit(similar[['area', 'build_year', 'object_type_code']].fillna(0))
distances, indices = nn.kneighbors([[area, build_year, type_code]])
analogs = similar.iloc[indices[0]]

# ============================================================
# Корректировки
# ============================================================
corrections = []
for _, analog in analogs.iterrows():
    corr = 1.0
    
    # Площадь
    if area > 100 and analog['area'] < 100:
        corr *= 0.95
    elif area < 50 and analog['area'] > 50:
        corr *= 1.05
    
    # Год постройки
    year_diff = build_year - analog['build_year']
    if abs(year_diff) > 5:
        corr *= 1 + (year_diff * 0.005)
    
    # Город
    if city and city not in str(analog.get('address', '')):
        corr *= 0.90
    
    # Вид разрешенного использования
    if permitted_use and permitted_use[:10].lower() not in str(analog.get('permitted_use', '')).lower():
        corr *= 0.95
    
    corrections.append(round(corr, 3))

# ============================================================
# Средневзвешенная цена
# ============================================================
total_weight = 0
weighted_sum = 0
for i, (_, analog) in enumerate(analogs.iterrows()):
    weight = 1 / (distances[0][i] + 0.01)
    adjusted_price = analog['price_per_sqm'] * corrections[i]
    weighted_sum += adjusted_price * weight
    total_weight += weight
weighted_avg_price = weighted_sum / total_weight if total_weight > 0 else price_sqm

# ============================================================
# Обоснование
# ============================================================
avg_analog = analogs['price_per_sqm'].mean()
diff_pct = (price_sqm - avg_analog) / avg_analog * 100

justification = f"""ОЦЕНКА ОБЪЕКТА{' с КН ' + kadastr if kadastr else ''}:
Тип: {object_type} | Площадь: {area:.0f} м² | Год: {build_year} | Город: {city if city else 'не определён'}
Вид разрешенного использования: {permitted_use if permitted_use else 'не указан'}

ЭТАП 1: ПРЕДВАРИТЕЛЬНЫЙ ОТБОР
Из базы {len(df)} сделок отобраны по критериям: тип={object_type}, площадь={area*0.3:.0f}-{area*3.0:.0f} м²"""

if city:
    justification += f", город={city}"
if permitted_use:
    justification += f", вид использования={permitted_use}"
justification += f"\nОтобрано: {len(similar)} объектов\n\nЭТАП 2: ФИНАЛЬНЫЙ ОТБОР 5 АНАЛОГОВ\n"

for i, (_, a) in enumerate(analogs.iterrows(), 1):
    kad = str(a.get('kadastr', ''))[:20]
    justification += f"Аналог {i}: {str(a.get('name',''))[:50]} | КН: {kad} | Площадь: {int(a['area'])} м² | Цена: {int(a['price_per_sqm'])} руб/м² | Корр: {corrections[i-1]:.3f}\n"

justification += f"""
ЭТАП 3: РАСЧЁТ
ML-прогноз: {price_sqm:.0f} руб/м² | Среднее аналогов: {avg_analog:.0f} руб/м² | Средневзвешенное: {weighted_avg_price:.0f} руб/м²
Финальная цена за м²: {price_sqm:.0f} руб/м² | Общая стоимость: {price_total:.0f} руб.

ЭТАП 4: ЗАКЛЮЧЕНИЕ
Рыночная стоимость определена в размере {price_total:.0f} руб. ({price_sqm:.0f} руб/м²). Отклонение от среднего аналогов: {diff_pct:+.1f}%."""

# ============================================================
# Итоговый JSON
# ============================================================
result = {
    "object": {
        "kadastr": kadastr,
        "area": area,
        "build_year": build_year,
        "object_type": object_type,
        "permitted_use": permitted_use,
        "city": city,
        "address": address
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
        "kadastr": str(a.get('kadastr', ''))[:20],
        "name": str(a.get('name', ''))[:80],
        "area": round(float(a['area']), 1),
        "price_per_sqm": round(float(a['price_per_sqm'])),
        "price_total": round(float(a.get('price_total', 0))),
        "build_year": int(a.get('build_year', 0)),
        "object_type": str(a.get('object_type', '')),
        "permitted_use": str(a.get('permitted_use', ''))[:50],
        "address": str(a.get('address', ''))[:120],
        "correction": corrections[i],
        "similarity": round(100 - distances[0][i] * 15, 1)
    })

print(json.dumps(result, ensure_ascii=False, indent=2))
