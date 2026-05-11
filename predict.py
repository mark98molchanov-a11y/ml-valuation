import pickle
import sys
import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Загружаем модель и данные
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("deals_clean.csv")

# Получаем параметры
area = float(sys.argv[1])
build_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
object_type = sys.argv[3] if len(sys.argv) > 3 else 'Помещение'

# Кодируем тип объекта
type_map = {
    'Земельный участок': 1,
    'Здание': 2,
    'Помещение': 3,
    'Сооружение': 4
}
type_code = type_map.get(object_type, 0)

# ============================================================
# 1. ML-прогноз цены
# ============================================================
price_sqm = model.predict([[area, build_year, type_code]])[0]
price_total = price_sqm * area

# ============================================================
# 2. Поиск 5 ближайших аналогов
# ============================================================
# Фильтруем похожие объекты
similar = df[
    (df['area'].between(area * 0.3, area * 3.0)) &
    (df['object_type_code'] == type_code)
].copy()

if len(similar) < 5:
    similar = df.copy()

# Подбираем аналоги
nn = NearestNeighbors(n_neighbors=min(5, len(similar)))
nn.fit(similar[['area', 'build_year', 'object_type_code']].fillna(0))
distances, indices = nn.kneighbors([[area, build_year, type_code]])

analogs = similar.iloc[indices[0]]

# Формируем результат
result = {
    "predicted": {
        "price_per_sqm": round(price_sqm),
        "price_total": round(price_total),
        "area": area,
        "build_year": build_year,
        "object_type": object_type
    },
    "analogs": []
}

for _, analog in analogs.iterrows():
    result["analogs"].append({
        "name": str(analog.get('name', analog.get('address', '')))[:100],
        "area": round(float(analog.get('area', 0)), 1),
        "price_per_sqm": round(float(analog.get('price_per_sqm', 0))),
        "price_total": round(float(analog.get('price_total', 0))),
        "build_year": int(analog.get('build_year', 0)),
        "object_type": str(analog.get('object_type', '')),
        "address": str(analog.get('address', ''))[:150]
    })

print(json.dumps(result, ensure_ascii=False))
