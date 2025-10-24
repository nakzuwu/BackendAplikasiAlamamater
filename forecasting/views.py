from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pandas as pd
from io import TextIOWrapper, BytesIO

def index(request):
    return render(request, 'forecasting/index.html')

def single_exponential_smoothing(data, alpha):
    """Metode Exponential Smoothing yang diperbaiki"""
    if len(data) == 0:
        return []
    
    forecast = [data[0]]  # Inisialisasi dengan data pertama
    for t in range(1, len(data)):
        # Rumus yang benar: F_t = α * A_(t-1) + (1-α) * F_(t-1)
        next_forecast = alpha * data[t-1] + (1 - alpha) * forecast[t-1]
        forecast.append(next_forecast)
    return forecast

def calculate_all_metrics(actual, forecast):
    """Menghitung MAPE, MAE, dan MSE"""
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Pastikan panjang data sama
    min_len = min(len(actual), len(forecast))
    actual = actual[:min_len]
    forecast = forecast[:min_len]
    
    # Hindari division by zero untuk MAPE
    non_zero_mask = actual != 0
    
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((actual[non_zero_mask] - forecast[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    mae = np.mean(np.abs(actual - forecast))
    mse = np.mean((actual - forecast) ** 2)
    
    return mape, mae, mse

def calculate_forecast_next(data, alpha, forecast_values):
    """Menghitung forecast untuk periode berikutnya"""
    if len(data) == 0 or len(forecast_values) == 0:
        return 0
    
    # Rumus: F_(t+1) = α * A_t + (1-α) * F_t
    last_actual = data[-1]
    last_forecast = forecast_values[-1]
    return alpha * last_actual + (1 - alpha) * last_forecast

def analyze_trend(data):
    """Menganalisis trend data"""
    if len(data) < 2:
        return "Stabil"
    
    # Hitung slope menggunakan linear regression sederhana
    x = np.arange(len(data))
    y = np.array(data)
    
    # Hitung slope
    n = len(x)
    slope = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / (n * np.sum(x*x) - np.sum(x)**2)
    
    if slope > 0.1:
        return "Naik"
    elif slope < -0.1:
        return "Turun"
    else:
        return "Stabil"

def process_input_data(request):
    """Process input data from either file upload or manual input"""
    df = None
    
    # 1. Input from CSV file
    if 'file' in request.FILES:
        file = request.FILES['file']
        df = pd.read_csv(TextIOWrapper(file.file, encoding='utf-8'))
    
    # 2. Input from manual textarea
    elif 'data' in request.POST and request.POST['data'].strip():
        df = process_manual_input(request.POST['data'])
    
    return df

def process_manual_input(data_text):
    """Process manual input data from textarea"""
    data_rows = []
    for row in data_text.split(';'):
        data_rows.append([x.strip() for x in row.split(',')])

    # Create flexible DataFrame
    headers = data_rows[0] if 'Tahun' in data_rows[0][0] else ['Tahun'] + [f'Ukuran{i}' for i in range(1, len(data_rows[0]))]
    df = pd.DataFrame(data_rows, columns=headers)

    # Convert data types
    df['Tahun'] = df['Tahun'].astype(int)
    for col in df.columns:
        if col != 'Tahun':
            df[col] = df[col].astype(float)
    
    return df

def extract_data_components(df):
    """Extract year column, years list, and size columns from DataFrame"""
    # Detect year column
    tahun_col = None
    for col in df.columns:
        if 'tahun' in col.lower():
            tahun_col = col
            break

    # Extract years
    if tahun_col:
        years = df[tahun_col].dropna().astype(int).tolist()
    else:
        years = list(range(1, len(df) + 1))

    # Extract size columns
    ukuran_cols = [c for c in df.columns if c != tahun_col]
    
    return tahun_col, years, ukuran_cols

def initialize_best_results():
    """Initialize best results tracking"""
    return {
        'alpha_mape': None,
        'mape': float('inf'),
        'forecast_mape': None,
        'alpha_mae': None,
        'mae': float('inf'),
        'alpha_mse': None,
        'mse': float('inf')
    }

def calculate_forecast_with_alpha(data, alpha):
    """Calculate forecast and metrics for a specific alpha"""
    forecast_values = single_exponential_smoothing(data, alpha)
    
    if len(data) > 1 and len(forecast_values) > 1:
        actual_eval = data[1:]
        pred_eval = forecast_values[1:]
        
        mape, mae, mse = calculate_all_metrics(actual_eval, pred_eval)
        
        # Calculate forecast for next period
        forecast_next = calculate_forecast_next(data, alpha, forecast_values)
        
        return {
            'alpha': alpha,
            'mape': round(mape, 2),
            'mae': round(mae, 2),
            'mse': round(mse, 2),
            'forecast_values': [round(x, 2) for x in forecast_values],
            'actual_values': data,
            'forecast_next': round(forecast_next, 2)
        }
    
    return None

def update_best_results(best_results, calculation, alpha):
    """Update best results if current calculation is better"""
    if calculation['mape'] < best_results['mape']:
        best_results['alpha_mape'] = alpha
        best_results['mape'] = calculation['mape']
        best_results['forecast_mape'] = calculation['forecast_values']
    
    if calculation['mae'] < best_results['mae']:
        best_results['alpha_mae'] = alpha
        best_results['mae'] = calculation['mae']
    
    if calculation['mse'] < best_results['mse']:
        best_results['alpha_mse'] = alpha
        best_results['mse'] = calculation['mse']

def analyze_forecast_trend(forecast_next, actual_last):
    """Analyze forecast trend compared to last actual value"""
    if forecast_next > actual_last:
        return "Naik"
    elif forecast_next < actual_last:
        return "Turun"
    else:
        return "Stabil"

def create_final_result(ukuran_name, data, years, best_results, calculations):
    """Create final result dictionary for a size"""
    tahun_last = years[-1]
    actual_last = data[-1]
    forecast_last = best_results['forecast_mape'][-1] if best_results['forecast_mape'] else 0
    next_year = tahun_last + 1
    forecast_next = calculate_forecast_next(data, best_results['alpha_mape'], best_results['forecast_mape'])
    
    # Analyze trends
    data_trend = analyze_trend(data)
    forecast_trend = analyze_forecast_trend(forecast_next, actual_last)
    
    result = {
        'ukuran': ukuran_name,
        'alpha': round(best_results['alpha_mape'], 1),
        'alpha_mape': round(best_results['alpha_mape'], 1),
        'alpha_mae': round(best_results['alpha_mae'], 1),
        'alpha_mse': round(best_results['alpha_mse'], 1),
        'mape': round(best_results['mape'], 2),
        'mae': round(best_results['mae'], 2),
        'mse': round(best_results['mse'], 2),
        'tahun_last': tahun_last,
        'actual_last': round(actual_last, 2),
        'forecast_last': round(forecast_last, 2),
        'tahun_next': next_year,
        'forecast_next': round(forecast_next, 2),
        'data_trend': data_trend,
        'forecast_trend': forecast_trend,
        'data_points': len(data),
        'data_range': f"{min(data)} - {max(data)}",
        'data_variance': round(np.var(data), 2) if len(data) > 1 else 0
    }
    
    chart_data = best_results['forecast_mape'] + [round(forecast_next, 2)] if best_results['forecast_mape'] else []
    
    return {
        'result': result,
        'chart_data': chart_data,
        'calculations': calculations
    }

def process_single_size_forecast(ukuran_name, data, years):
    """Process forecasting for a single size"""
    # Test all alpha values
    calculations = []
    best_results = initialize_best_results()
    
    for alpha in [i/10 for i in range(1, 10)]:
        calculation = calculate_forecast_with_alpha(data, alpha)
        if calculation:
            calculations.append(calculation)
            update_best_results(best_results, calculation, alpha)
    
    # Create final result if valid forecasts exist
    if best_results['forecast_mape']:
        result = create_final_result(
            ukuran_name, data, years, best_results, calculations
        )
        return result
    
    return None

def process_forecasting(ukuran_cols, df, tahun_col, years):
    """Process forecasting for all size columns"""
    results = []
    chart_data = {}
    all_calculations = {}
    
    for col in ukuran_cols:
        try:
            data = df[col].dropna().astype(float).tolist()
            if len(data) < 2:
                continue
                
            result = process_single_size_forecast(col, data, years)
            if result:
                results.append(result['result'])
                chart_data[col] = result['chart_data']
                all_calculations[col] = result['calculations']
                
        except (ValueError, TypeError):
            continue
    
    return results, chart_data, all_calculations

def prepare_context(results, chart_data, years, all_calculations):
    """Prepare context for template rendering"""
    best_mape_all = min([r['mape'] for r in results]) if results else None
    total_calculations = sum(len(calcs) for calcs in all_calculations.values())
    
    mae_best_results = []
    mse_best_results = []
    
    for ukuran, calculations in all_calculations.items():
        # Find best MAE for this ukuran
        if calculations:
            best_mae_calc = min(calculations, key=lambda x: x['mae'])
            # Find corresponding original result for additional data
            original_result = next((r for r in results if r['ukuran'] == ukuran), None)
            if original_result:
                mae_best_results.append({
                    'ukuran': ukuran,
                    'alpha': best_mae_calc['alpha'],
                    'mape': best_mae_calc['mape'],  # Pastikan ini ada
                    'mae': best_mae_calc['mae'],    # Pastikan ini ada  
                    'mse': best_mae_calc['mse'],    # Pastikan ini ada
                    'forecast_next': best_mae_calc['forecast_next'],
                    'forecast_last': best_mae_calc['forecast_values'][-1] if best_mae_calc['forecast_values'] else original_result['actual_last'],
                    'tahun_last': original_result['tahun_last'],
                    'actual_last': original_result['actual_last'],
                    'tahun_next': original_result['tahun_next'],
                    'forecast_trend': analyze_forecast_trend(best_mae_calc['forecast_next'], original_result['actual_last'])
                })
            
            # Find best MSE for this ukuran
            best_mse_calc = min(calculations, key=lambda x: x['mse'])
            if original_result:
                mse_best_results.append({
                    'ukuran': ukuran,
                    'alpha': best_mse_calc['alpha'],
                    'mape': best_mse_calc['mape'],  # Pastikan ini ada
                    'mae': best_mse_calc['mae'],    # Pastikan ini ada
                    'mse': best_mse_calc['mse'],    # Pastikan ini ada
                    'forecast_next': best_mse_calc['forecast_next'],
                    'forecast_last': best_mse_calc['forecast_values'][-1] if best_mse_calc['forecast_values'] else original_result['actual_last'],
                    'tahun_last': original_result['tahun_last'],
                    'actual_last': original_result['actual_last'],
                    'tahun_next': original_result['tahun_next'],
                    'forecast_trend': analyze_forecast_trend(best_mse_calc['forecast_next'], original_result['actual_last'])
                })
    
    return {
        'results': results,
        'chart_data': chart_data,
        'years': years,
        'best_mape': best_mape_all,
        'all_calculations': all_calculations,
        'total_calculations': total_calculations,
        'mae_best_results': mae_best_results,
        'mse_best_results': mse_best_results
    }

def forecast(request):
    if request.method == 'POST':
        try:
            # Process input data
            df = process_input_data(request)
            if df is None:
                return render(request, 'forecasting/result.html', {
                    'error': 'Harap upload CSV atau masukkan data manual.'
                })

            # Extract data components
            tahun_col, years, ukuran_cols = extract_data_components(df)
            if not ukuran_cols:
                return render(request, 'forecasting/result.html', {
                    'error': 'Tidak ada kolom ukuran yang valid.'
                })

            # Process forecasting for all sizes
            results, chart_data, all_calculations = process_forecasting(ukuran_cols, df, tahun_col, years)
            
            # Prepare context
            context = prepare_context(results, chart_data, years, all_calculations)
            
            return render(request, 'forecasting/result.html', context)

        except Exception as e:
            return render(request, 'forecasting/result.html', {
                'error': f'Terjadi kesalahan: {str(e)}'
            })

    return render(request, 'forecasting/index.html')

# =========================================================
# 🌐 API ENDPOINT: /api/forecast/
# =========================================================
@csrf_exempt
def api_forecast(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Gunakan metode POST.'}, status=405)

    try:
        df = process_input_data(request)
        if df is None:
            return JsonResponse({'error': 'Harap kirim file CSV/Excel atau JSON valid.'}, status=400)

        # Extract data components
        tahun_col, years, ukuran_cols = extract_data_components(df)
        if not ukuran_cols:
            return JsonResponse({'error': 'Tidak ditemukan kolom ukuran.'}, status=400)

        # Process forecasting for all sizes
        results, chart_data, all_calculations = process_forecasting(ukuran_cols, df, tahun_col, years)
        
        best_mape_all = min([r['mape'] for r in results]) if results else None
        
        # Update years with next year if needed
        next_year = None
        if results and 'tahun_next' in results[0]:
            next_year = results[0]['tahun_next']
            if next_year and next_year not in years:
                years.append(next_year)

        return JsonResponse({
            'status': 'success',
            'results': results,
            'best_mape': best_mape_all,
            'chart_data': chart_data,
            'years': years,
            'all_calculations': all_calculations,
            'total_calculations': sum(len(calcs) for calcs in all_calculations.values())
        }, status=200, json_dumps_params={'ensure_ascii': False})

    except Exception as e:
        return JsonResponse({'error': f'Gagal memproses data: {str(e)}'}, status=400)