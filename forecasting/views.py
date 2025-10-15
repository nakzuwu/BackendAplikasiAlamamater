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

    # 1. Input via file
    if 'file' in request.FILES:
        file = request.FILES['file']
        filename = file.name.lower()
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(TextIOWrapper(file.file, encoding='utf-8'))
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(BytesIO(file.read()))
            else:
                return JsonResponse({'error': 'Format file tidak didukung. Gunakan CSV atau Excel.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Gagal membaca file: {str(e)}'}, status=400)

    # 2. Input manual JSON
    elif request.content_type == 'application/json':
        import json
        try:
            body = json.loads(request.body)
            df = pd.DataFrame(body)
        except Exception as e:
            return JsonResponse({'error': f'Gagal membaca JSON: {str(e)}'}, status=400)
    else:
        return JsonResponse({'error': 'Harap kirim file CSV/Excel atau JSON valid.'}, status=400)

    # 3. Deteksi kolom tahun
    tahun_col = None
    for col in df.columns:
        if "tahun" in col.lower():
            tahun_col = col
            break

    if tahun_col:
        years = df[tahun_col].dropna().astype(int).tolist()
    else:
        years = list(range(1, len(df) + 1))

    # 4. Ambil semua kolom ukuran otomatis (selain tahun)
    ukuran_cols = [c for c in df.columns if c != tahun_col]
    if not ukuran_cols:
        return JsonResponse({'error': 'Tidak ditemukan kolom ukuran.'}, status=400)

    # 5. Proses forecast
    results = []
    chart_data = {}
    next_year = None

    for col in ukuran_cols:
        try:
            data = df[col].dropna().astype(float).tolist()
        except:
            continue
        if len(data) < 2:  # minimal 2 data
            continue

        best_alpha = None
        best_mape = float('inf')
        best_forecast = None

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

    best_mape_all = min([r['mape'] for r in results]) if results else None
    if next_year and next_year not in years:
        years.append(next_year)

    return JsonResponse({
        'status': 'success',
        'results': results,
        'best_mape': best_mape_all,
        'chart_data': chart_data,
        'years': years
    }, status=200, json_dumps_params={'ensure_ascii': False})


def forecast(request):
    if request.method == 'POST':
        import pandas as pd
        from io import TextIOWrapper, StringIO

        # 1. Input CSV
        if 'file' in request.FILES:
            file = request.FILES['file']
            df = pd.read_csv(TextIOWrapper(file.file, encoding='utf-8'))

        # 2. Input manual (textarea)
        elif 'data' in request.POST and request.POST['data'].strip():
            data_text = request.POST['data']
            data_rows = []
            for row in data_text.split(';'):
                data_rows.append([x.strip() for x in row.split(',')])

            # Buat DataFrame fleksibel
            headers = data_rows[0] if 'Tahun' in data_rows[0][0] else ['Tahun'] + [f'Ukuran{i}' for i in range(1, len(data_rows[0]))]
            df = pd.DataFrame(data_rows, columns=headers)

            # Konversi tipe data (tahun=int, ukuran=float)
            df['Tahun'] = df['Tahun'].astype(int)
            for col in df.columns:
                if col != 'Tahun':
                    df[col] = df[col].astype(float)
        else:
            return render(request, 'forecasting/result.html', {
                'error': 'Harap upload CSV atau masukkan data manual.'
            })

        # ===== Deteksi kolom tahun =====
        tahun_col = None
        for col in df.columns:
            if 'tahun' in col.lower():
                tahun_col = col
                break

        if tahun_col:
            years = df[tahun_col].dropna().astype(int).tolist()
        else:
            years = list(range(1, len(df) + 1))

        # ===== Ambil semua kolom ukuran otomatis =====
        ukuran_cols = [c for c in df.columns if c != tahun_col]
        if not ukuran_cols:
            return render(request, 'forecasting/result.html', {'error': 'Tidak ada kolom ukuran.'})

        # ===== Forecast =====
        results = []
        chart_data = {}
        next_year = None

        for col in ukuran_cols:
            try:
                data = df[col].dropna().astype(float).tolist()
            except:
                continue
            if len(data) < 2:
                continue

            best_alpha = None
            best_mape = float('inf')
            best_forecast = None

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

        if next_year and next_year not in years:
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
