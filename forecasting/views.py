from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pandas as pd
from io import TextIOWrapper, BytesIO

def index(request):
    return render(request, 'forecasting/index.html')

def single_exponential_smoothing(data, alpha):
    forecast = [data[0]]
    for t in range(1, len(data)):
        next_forecast = alpha * data[t-1] + (1 - alpha) * forecast[-1]
        forecast.append(next_forecast)
    return forecast

def calculate_mape(actual, forecast):
    actual = np.array(actual)
    forecast = np.array(forecast)
    return np.mean(np.abs((actual - forecast) / actual)) * 100

# =========================================================
# 🌐 API ENDPOINT: /api/forecast/
# =========================================================
@csrf_exempt
def api_forecast(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Gunakan metode POST.'}, status=405)

    df = None

    # 🧩 1. Input via file (CSV / Excel)
    if 'file' in request.FILES:
        file = request.FILES['file']
        filename = file.name.lower()

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(TextIOWrapper(file.file, encoding='utf-8'))
            elif filename.endswith('.xls') or filename.endswith('.xlsx'):
                df = pd.read_excel(BytesIO(file.read()))
            else:
                return JsonResponse({'error': 'Format file tidak didukung. Gunakan CSV atau Excel.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Gagal membaca file: {str(e)}'}, status=400)

    # 🧩 2. Input manual JSON
    elif request.content_type == 'application/json':
        try:
            import json
            body = json.loads(request.body)
            df = pd.DataFrame(body)
        except Exception as e:
            return JsonResponse({'error': f'Gagal membaca JSON: {str(e)}'}, status=400)
    else:
        return JsonResponse({'error': 'Harap kirim file CSV/Excel atau JSON valid.'}, status=400)

    # 🧠 3. Deteksi kolom tahun
    tahun_col = None
    for col in df.columns:
        if "tahun" in col.lower():
            tahun_col = col
            break

    if tahun_col:
        years = df[tahun_col].dropna().astype(int).tolist()
    else:
        years = list(range(1, len(df) + 1))

    # 🧠 4. Filter kolom ukuran jas
    ukuran_cols = [c for c in df.columns if c != tahun_col and c.upper() in ['S', 'M', 'L', 'XL', 'XXL']]
    if not ukuran_cols:
        return JsonResponse({'error': 'Tidak ditemukan kolom ukuran (S, M, L, XL, XXL).'}, status=400)

    # 🔢 5. Proses per kolom
    results = []
    chart_data = {}

    for col in ukuran_cols:
        data = df[col].dropna().astype(float).tolist()
        if len(data) < 3:
            continue

        best_alpha = None
        best_mape = float('inf')
        best_forecast = None

        # Cari alpha terbaik
        for alpha in [i/10 for i in range(1, 10)]:
            forecast_values = single_exponential_smoothing(data, alpha)
            actual = data[1:]
            pred = forecast_values[1:]
            mape = calculate_mape(actual, pred)
            if mape < best_mape:
                best_mape = mape
                best_alpha = alpha
                best_forecast = forecast_values

        tahun_last = years[-1]
        actual_last = data[-1]
        forecast_last = best_forecast[-1]
        next_year = tahun_last + 1
        forecast_next = best_alpha * actual_last + (1 - best_alpha) * forecast_last

        results.append({
            'ukuran': col,
            'alpha': round(best_alpha, 1),
            'mape': round(best_mape, 2),
            'tahun_last': tahun_last,
            'actual_last': round(actual_last, 2),
            'forecast_last': round(forecast_last, 2),
            'tahun_next': next_year,
            'forecast_next': round(forecast_next, 2)
        })

        chart_data[col] = [round(x, 2) for x in best_forecast] + [round(forecast_next, 2)]

    # 🎯 TAMBAHKAN BEST_MAPE DI SINI!
    if results:
        best_mape_all = min([r['mape'] for r in results])
        # Tambah tahun prediksi ke chart
        if next_year not in years:
            years.append(next_year)
    else:
        best_mape_all = None

    return JsonResponse({
        'status': 'success',
        'results': results,
        'best_mape': best_mape_all,  
        'chart_data': chart_data,
        'years': years
    }, status=200, json_dumps_params={'ensure_ascii': False})

def forecast(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return render(request, 'forecasting/result.html', {
                'error': 'Harap upload file CSV yang berisi kolom ukuran jas.'
            })

        file = request.FILES['file']
        df = pd.read_csv(TextIOWrapper(file.file, encoding='utf-8'))
        print("=== Kolom CSV ===")
        print(df.columns.tolist())


        # Deteksi kolom tahun
        tahun_col = None
        for col in df.columns:
            if "tahun" in col.lower():
                tahun_col = col
                break

        if tahun_col:
            years = df[tahun_col].dropna().astype(int).tolist()
        else:
            years = list(range(1, len(df) + 1))

        # Filter kolom ukuran jas
        ukuran_cols = [c for c in df.columns if c != tahun_col and c.upper() in ['S','M','L','XL','XXL']]

        if not ukuran_cols:
            return render(request, 'forecasting/result.html', {
                'error': 'Harap upload file CSV yang berisi kolom S, M, L, XL, dan XXL.'
            })

        results = []
        chart_data = {}

        for col in ukuran_cols:
            data = df[col].dropna().astype(float).tolist()

            if len(data) < 3:
                continue

            best_alpha = None
            best_mape = float('inf')
            best_forecast = None

            # 🔍 Cari alpha terbaik
            for alpha in [i/10 for i in range(1, 10)]:
                forecast_values = single_exponential_smoothing(data, alpha)
                actual = data[1:]
                pred = forecast_values[1:]
                mape = calculate_mape(actual, pred)
                if mape < best_mape:
                    best_mape = mape
                    best_alpha = alpha
                    best_forecast = forecast_values

            tahun_last = years[-1]
            actual_last = data[-1]
            forecast_last = best_forecast[-1]
            next_year = tahun_last + 1
            forecast_next = best_alpha * actual_last + (1 - best_alpha) * forecast_last

            # Tambah ke hasil
            results.append({
                'ukuran': col,
                'alpha': round(best_alpha, 1),
                'mape': round(best_mape, 2),
                'tahun_last': tahun_last,
                'actual_last': round(actual_last, 2),
                'forecast_last': round(forecast_last, 2),
                'tahun_next': next_year,
                'forecast_next': round(forecast_next, 2)
            })

            chart_data[col] = [round(x, 2) for x in best_forecast] + [round(forecast_next, 2)]

        # Tambah tahun prediksi ke chart
        if next_year not in years:
            years.append(next_year)

        best_mape_all = min([r['mape'] for r in results]) if results else None

        context = {
            'results': results,
            'chart_data': chart_data,
            'years': years,
            'best_mape': best_mape_all,
        }

        return render(request, 'forecasting/result.html', context)

    return render(request, 'forecasting/index.html')
