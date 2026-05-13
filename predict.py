import pickle
import sys
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def is_empty(val):
    if pd.isna(val): return True
    s = str(val).strip().lower()
    return s in ['', 'nan', 'none', 'null', '-', 'нет']

def clean_val(val, max_len=80):
    if pd.isna(val) or str(val).strip().lower() in ['nan', 'none', '']: return ''
    return str(val).strip()[:max_len]

# Загружаем модели
try:
    with open("model_buildings.pkl", "rb") as f: model_buildings = pickle.load(f)
except: model_buildings = None
try:
    with open("model_land.pkl", "rb") as f: model_land = pickle.load(f)
except: model_land = None

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
        for word in addr.replace(',', ' ').replace('.', ' ').split():
            if word[0].isupper() and len(word) > 3: all_cities.add(word)
    for part in address.replace(',', ' ').split():
        if part in all_cities: city = part; break
    if not city:
        for c in ['Салехард','Новый','Ноябрьск','Тарко-Сале','Надым','Губкинский','Муравленко','Лабытнанги','Красноселькуп']:
            if c in address: city = c; break

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

if is_land and permitted_use:
    similar = similar[~similar['permitted_use'].apply(is_empty)].copy()
    kw = permitted_use.lower().split()
    uf = similar[similar['permitted_use'].str.lower().apply(lambda x: any(k in x for k in kw) if pd.notna(x) else False)]
    similar = uf if len(uf) >= 3 else similar[similar['permitted_use'].str.lower().str.contains(kw[0], na=False)] if kw and len(similar[similar['permitted_use'].str.lower().str.contains(kw[0], na=False)]) >= 3 else similar

if not is_land:
    if object_name:
        kw = object_name.lower().split()
        nf = similar[similar['name'].apply(lambda x: sum(1 for k in kw if k in str(x).lower()) >= min(2,len(kw)) if pd.notna(x) else False)]
        similar = nf if len(nf) >= 3 else similar[similar['name'].astype(str).str.lower().str.contains(kw[0], na=False)] if kw and len(similar[similar['name'].astype(str).str.lower().str.contains(kw[0], na=False)]) >= 3 else similar
    if wall_material:
        kw = wall_material.lower().split()
        mf = similar[similar['wall_material'].astype(str).str.lower().apply(lambda x: any(k in x for k in kw) if pd.notna(x) else False)]
        if len(mf) >= 3: similar = mf

if city and len(similar) >= 5:
    cf = similar[similar['address'].str.contains(city, na=False)]
    if len(cf) >= 3: similar = cf

if len(similar) < 5: similar = df[df['object_type_code'] == type_code].copy()
if len(similar) < 3: similar = df.copy()

# ============================================================
# Поиск 5 ближайших
# ============================================================
n_neighbors = min(5, len(similar))
nn = NearestNeighbors(n_neighbors=n_neighbors)
scaler = StandardScaler()

if is_land:
    similar['use_code'] = pd.factorize(similar['permitted_use'])[0]
    feats = ['area', 'build_year', 'use_code']
    fs = scaler.fit_transform(similar[feats].fillna(0)); fs[:,2] *= 3
    nn.fit(fs)
    uc = pd.factorize(df['permitted_use'])[0][df['permitted_use'] == permitted_use]; uc = uc[0] if len(uc) > 0 else 0
    os_ = scaler.transform([[area, build_year, uc]]); os_[:,2] *= 3
    distances, indices = nn.kneighbors(os_)
    search_desc = "площадь, год, ВРИ (×3)"
else:
    similar['name_code'] = pd.factorize(similar['name'])[0]
    similar['material_code'] = pd.factorize(similar['wall_material'])[0]
    feats = ['area', 'build_year', 'object_type_code', 'name_code', 'material_code']
    fs = scaler.fit_transform(similar[feats].fillna(0)); fs[:,3] *= 5; fs[:,4] *= 3
    nn.fit(fs)
    # Ищем коды внутри similar
    nc = 0
    for code, name in enumerate(pd.factorize(similar['name'])[1]):
        if object_name[:10].lower() in str(name).lower(): nc = code; break
    mc = 0
    for code, mat in enumerate(pd.factorize(similar['wall_material'])[1]):
        if wall_material[:10].lower() in str(mat).lower(): mc = code; break
    os_ = scaler.transform([[area, build_year, type_code, nc, mc]]); os_[:,3] *= 5; os_[:,4] *= 3
    distances, indices = nn.kneighbors(os_)
    search_desc = "площадь, год, тип, наименование (×5), материал (×3)"

analogs = similar.iloc[indices[0]]

# Корректировки
corrections = []
for _, a in analogs.iterrows():
    c = 1.0
    if area > 100 and a['area'] < 100: c *= 0.95
    elif area < 50 and a['area'] > 50: c *= 1.05
    yd = build_year - a['build_year']
    if abs(yd) > 5: c *= 1 + (yd * 0.005)
    if city and city not in str(a.get('address', '')): c *= 0.90
    corrections.append(round(c, 3))

# Средневзвешенная
tw, ws = 0, 0
for i, (_, a) in enumerate(analogs.iterrows()):
    w = 1/(distances[0][i]+0.01)
    ws += a['price_per_sqm'] * corrections[i] * w
    tw += w
wap = ws/tw if tw > 0 else price_sqm
aa = analogs['price_per_sqm'].mean()
dp = (price_sqm - aa)/aa * 100

# Обоснование
j = f"""ОЦЕНКА ОБЪЕКТА{' с КН '+kadastr if kadastr else ''}:
Тип: {object_type} | Площадь: {area:.0f} м² | Год: {build_year} | Город: {city if city else 'не определён'}"""
if is_land and permitted_use: j += f"\nВРИ: {permitted_use}"
if not is_land:
    if object_name: j += f"\nНаименование: {object_name}"
    if wall_material: j += f"\nМатериал стен: {wall_material}"
j += f"""

ЭТАП 1: ПРЕДВАРИТЕЛЬНЫЙ ОТБОР
Из базы {len(df)} сделок отобраны: тип={object_type}, площадь={area*0.3:.0f}-{area*3.0:.0f} м²"""
if city: j += f", город={city}"
if is_land and permitted_use: j += f", ВРИ похожие на «{permitted_use}»"
if not is_land and object_name: j += f", наименование похожее на «{object_name}»"
if not is_land and wall_material: j += f", материал «{wall_material}»"
j += f"\nОтобрано: {len(similar)} объектов\n\nЭТАП 2: ФИНАЛЬНЫЙ ОТБОР 5 АНАЛОГОВ (по {search_desc})\n"

for i, (_, a) in enumerate(analogs.iterrows(), 1):
    yr = int(a.get('build_year',0)) if not pd.isna(a.get('build_year')) else 0
    j += f"Аналог {i}: {clean_val(a.get('name',''),50)} | КН: {clean_val(a.get('kadastr',''),20)} | Площадь: {int(a['area'])} м² | Год: {yr} | Цена: {int(a['price_per_sqm'])} руб/м²"
    if clean_val(a.get('wall_material',''),15): j += f" | Материал: {clean_val(a.get('wall_material',''),15)}"
    if clean_val(a.get('permitted_use',''),30): j += f" | ВРИ: {clean_val(a.get('permitted_use',''),30)}"
    j += f" | Корр: {corrections[i-1]:.3f}\n"

j += f"""
ЭТАП 3: РАСЧЁТ
ML-прогноз: {price_sqm:.0f} руб/м² | Среднее аналогов: {aa:.0f} руб/м² | Средневзвешенное: {wap:.0f} руб/м²
Финальная цена: {price_sqm:.0f} руб/м² | Общая стоимость: {price_total:.0f} руб.

ЭТАП 4: ЗАКЛЮЧЕНИЕ
Рыночная стоимость: {price_total:.0f} руб. ({price_sqm:.0f} руб/м²). Отклонение от аналогов: {dp:+.1f}%."""

# JSON
result = {
    "object": {"kadastr":clean_val(kadastr,20),"area":area,"build_year":build_year,"object_type":object_type,
               "permitted_use":clean_val(permitted_use,50),"name":clean_val(object_name,80),
               "wall_material":clean_val(wall_material,30),"city":clean_val(city,30),"address":clean_val(address,120)},
    "predicted": {"price_per_sqm":round(price_sqm),"price_total":round(price_total)},
    "calculation": {"ml_prediction":round(price_sqm),"avg_analogs":round(aa),"weighted_avg":round(wap),"deviation_pct":round(dp,1)},
    "justification": j, "analogs": [], "search_features": search_desc
}

for i, (_, a) in enumerate(analogs.iterrows()):
    yr = int(a.get('build_year',0)) if not pd.isna(a.get('build_year')) else 0
    result["analogs"].append({
        "num":i+1,"kadastr":clean_val(a.get('kadastr',''),20),"name":clean_val(a.get('name',''),80),
        "area":round(float(a['area']),1),"price_per_sqm":round(float(a['price_per_sqm'])),
        "price_total":round(float(a.get('price_total',0))),"build_year":yr,
        "object_type":clean_val(a.get('object_type',''),30),"permitted_use":clean_val(a.get('permitted_use',''),50),
        "wall_material":clean_val(a.get('wall_material',''),30),"address":clean_val(a.get('address',''),120),
        "correction":corrections[i],"similarity":round(100-distances[0][i]*15,1)
    })

print(json.dumps(result, ensure_ascii=False, indent=2))
