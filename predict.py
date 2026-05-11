import pickle
import sys
import json

# Загружаем модель
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Получаем параметры
area = float(sys.argv[1])
build_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2015

# Прогноз
price_sqm = model.predict([[area, build_year]])[0]
price_total = price_sqm * area

result = {
    "price_per_sqm": round(price_sqm),
    "price_total": round(price_total),
    "area": area,
    "build_year": build_year
}

print(json.dumps(result, ensure_ascii=False))
