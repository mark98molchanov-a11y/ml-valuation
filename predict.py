import pickle
import sys
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Загружаем модель и данные
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("deals_clean.csv")

# Получаем параметры
area = float(sys.argv[1])
build_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
object_type = sys.argv[3] if len(sys.argv) > 3 else 'Помещение'
usage = sys.argv[4] if len(sys.argv) > 4 else ''
address = sys.argv[5] if len(sys.argv) > 5 else ''

# Кодируем тип
type_map = {'Земельный участок': 1, 'Здание': 2, 'Помещение': 3, 'Сооружение': 4}
type_code = type_map.get(object_type, 0)

# Город из адреса
city = ''
for c in ['Салехард', 'Новый Уренгой', 'Ноябрьск', 'Тарко-Сале', 'Надым']:
    if c in address:
        city = c
        break

# ============================================================
# 1. ML-прогноз
# ============================================================
price_sqm = model.predict([[area, build_year, type_code]])[0]
price_total = price_sqm * area

# ============================================================
# 2. Подбор аналогов
# ============================================================
# Фильтр 1: тот же тип
similar = df[df['object_type_code'] == type_code].copy()

# Фильтр 2: похожая площадь
similar = similar[similar['area'].between(area * 0.3, area * 3.0)]

# Фильтр 3: город (если есть)
if city:
    city_filtered = similar[similar['address'].str.contains(city, na=False)]
    if len(city_filtered) >= 3:
        similar = city_filtered

# Фильтр 4: вид использования
if usage:
    usage_filtered = similar[similar['usage'].str.contains(usage[:15], na=False)]
    if len(usage_filtered) >= 3:
        similar = usage_filtered

# Если слишком мало — расширяем
if len(similar) < 5:
    similar = df[df['object_type_code'] == type_code].copy()

# Поиск 5 ближайших
nn = NearestNeighbors(n_neighbors=min(5, len(similar)))
nn.fit(similar[['area', 'build_year', 'object_type_code']].fillna(0))
distances, indices = nn.kneighbors([[area, build_year, type_code]])
analogs = similar.iloc[indices[0]]

# ============================================================
# 3. Расчёт корректировок
# ============================================================
corrections = []
for _, analog in analogs.iterrows():
    corr = 1.0
    
    # Корректировка на площадь
    if area > 100 and analog['area'] < 100:
        corr *= 0.95  # Скидка за большой размер
    elif area < 50 and analog['area'] > 50:
        corr *= 1.05  # Надбавка за маленький размер
    
    # Корректировка на год постройки
    year_diff = build_year - analog['build_year']
    if abs(year_diff) > 5:
        corr *= 1 + (year_diff * 0.005)  # 0.5% за год
    
    # Корректировка на город
    if city and city not in str(analog.get('address', '')):
        corr *= 0.90  # Скидка за менее престижный город
    
    corrections.append(round(corr, 3))

# ============================================================
# 4. Расчёт средневзвешенной цены
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
# 5. Формируем обоснование
# ============================================================
justification_parts = []

# Заголовок
justification_parts.append(f"ОЦЕНКА ОБЪЕКТА:")
justification_parts.append(f"Тип: {object_type} | Площадь: {area:.0f} м² | Год: {build_year} | Город: {city if city else 'не определён'}")
justification_parts.append("")

# Этап 1: Отбор
justification_parts.append("ЭТАП 1: ПРЕДВАРИТЕЛЬНЫЙ ОТБОР")
justification_parts.append(f"Из базы {len(df)} сделок отобраны объекты по критериям:")
justification_parts.append(f"- Тип: {object_type}")
justification_parts.append(f"- Площадь: {area*0.3:.0f}-{area*3.0:.0f} м²")
if city:
    justification_parts.append(f"- Город: {city}")
if usage:
    justification_parts.append(f"- Использование: {usage}")
justification_parts.append(f"Отобрано: {len(similar)} объектов")
justification_parts.append("")

# Этап 2: Аналоги
justification_parts.append("ЭТАП 2: ФИНАЛЬНЫЙ ОТБОР 5 АНАЛОГОВ")
for i, (_, analog) in enumerate(analogs.iterrows(), 1):
    justification_parts.append(
        f"Аналог {i}: {str(analog.get('name', ''))[:50]} | "
        f"Площадь: {analog['area']:.0f} м² | "
        f"Цена: {analog['price_per_sqm']:.0f} руб/м² | "
        f"Корректировка: {corrections[i-1]:.3f}"
    )
justification_parts.append("")

# Этап 3: Расчёт
justification_parts.append("ЭТАП 3: РАСЧЁТ СТОИМОСТИ")
justification_parts.append(f"ML-прогноз (CatBoost): {price_sqm:.0f} руб/м²")
justification_parts.append(f"Среднее аналогов: {analogs['price_per_sqm'].mean():.0f} руб/м²")
justification_parts.append(f"Средневзвешенное с корректировками: {weighted_avg_price:.0f} руб/м²")
justification_parts.append(f"Финальная цена за м²: {price_sqm:.0f} руб/м²")
justification_parts.append(f"Общая стоимость: {price_total:.0f} руб.")
justification_parts.append("")

# Этап 4: Вывод
avg_analog = analogs['price_per_sqm'].mean()
diff_pct = (price_sqm - avg_analog) / avg_analog * 100
justification_parts.append("ЭТАП 4: ЗАКЛЮЧЕНИЕ")
justification_parts.append(f"Рыночная стоимость объекта определена в размере {price_total:.0f} руб. "
                          f"({price_sqm:.0f} руб/м²) на основе сравнительного подхода. "
                          f"Отклонение от среднего аналогов: {diff_pct:+.1f}%. "
                          f"Учтены корректировки на площадь, год постройки и местоположение.")

justification = "\n".join(justification_parts)

# ============================================================
# 6. Итоговый JSON
# ============================================================
result = {
    "predicted": {
        "price_per_sqm": round(price_sqm),
        "price_total": round(price_total),
        "area": area,
        "build_year": build_year,
        "object_type": object_type,
        "usage": usage,
        "city": city
    },
    "calculation": {
        "ml_prediction": round(price_sqm),
        "avg_analogs": round(analogs['price_per_sqm'].mean()),
        "weighted_avg": round(weighted_avg_price),
        "deviation_pct": round(diff_pct, 1)
    },
    "justification": justification,
    "analogs": []
}

for i, (_, analog) in enumerate(analogs.iterrows()):
    result["analogs"].append({
        "num": i + 1,
        "name": str(analog.get('name', ''))[:80],
        "area": round(float(analog.get('area', 0)), 1),
        "price_per_sqm": round(float(analog.get('price_per_sqm', 0))),
        "price_total": round(float(analog.get('price_total', 0))),
        "build_year": int(analog.get('build_year', 0)),
        "object_type": str(analog.get('object_type', '')),
        "usage": str(analog.get('usage', ''))[:50],
        "address": str(analog.get('address', ''))[:120],
        "correction": corrections[i],
        "similarity": round(100 - distances[0][i] * 15, 1)
    })

print(json.dumps(result, ensure_ascii=False, indent=2))
