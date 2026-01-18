from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pandas as pd
from io import TextIOWrapper, BytesIO
import json
from decimal import Decimal, ROUND_HALF_UP

def index(request):
    return render(request, 'forecasting/index.html')

def single_exponential_smoothing(data, alpha):
    """Metode Exponential Smoothing yang diperbaiki"""
    if len(data) == 0:
        return []
    
    forecast = [data[0]]  # Inisialisasi dengan data pertama
    for t in range(1, len(data)):
        next_forecast = alpha * data[t-1] + (1 - alpha) * forecast[t-1]
        forecast.append(next_forecast)
    return forecast

def calculate_all_metrics(actual, forecast):
    """Menghitung MAPE, MAE, dan MSE"""
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    min_len = min(len(actual), len(forecast))
    actual = actual[:min_len]
    forecast = forecast[:min_len]
    
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
    
    last_actual = data[-1]
    last_forecast = forecast_values[-1]
    return alpha * last_actual + (1 - alpha) * last_forecast

def analyze_trend(data):
    """Menganalisis trend data"""
    if len(data) < 2:
        return "Stabil"
    
    x = np.arange(len(data))
    y = np.array(data)
    
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
    """Process manual input data from textarea dengan format baru"""
    lines = [line.strip() for line in data_text.strip().split('\n') if line.strip()]
    
    if len(lines) < 2:
        raise ValueError("Format data tidak valid. Minimal 2 baris.")
    
    # Coba format baru: baris pertama header ukuran
    if ',' in lines[0]:
        ukuran_headers = [x.strip() for x in lines[0].split(',')]
        data_rows = []
        tahun_list = []
        
        for line in lines[1:]:
            parts = [x.strip() for x in line.split(',')]
            
            # Pastikan jumlah data sesuai
            if len(parts) == len(ukuran_headers) + 1:  # +1 untuk tahun
                tahun_list.append(int(parts[0]))
                data_rows.append([float(x) for x in parts[1:]])
            elif len(parts) == len(ukuran_headers):  # jika tahun tidak ada
                tahun_list.append(len(tahun_list) + 1)  # generate tahun otomatis
                data_rows.append([float(x) for x in parts])
            else:
                # Coba format lama
                break
        
        if data_rows:
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=ukuran_headers)
            df.insert(0, 'Tahun', tahun_list)
            return df
    
    # Jika format baru gagal, coba format lama
    data_rows = []
    for row in data_text.split(';'):
        if row.strip():
            row_data = [x.strip() for x in row.split(',')]
            if row_data:
                # Coba konversi ke numerik
                converted_row = []
                for item in row_data:
                    try:
                        converted_row.append(float(item))
                    except ValueError:
                        converted_row.append(item)
                data_rows.append(converted_row)

    if not data_rows:
        raise ValueError("Data kosong atau format tidak valid.")

    # Coba deteksi header
    first_row = data_rows[0]
    is_header_numeric = all(isinstance(x, (int, float)) for x in first_row)
    
    if not is_header_numeric and len(data_rows) > 1:
        # Baris pertama adalah header
        headers = ['Tahun'] + [str(x) for x in first_row[1:]] if 'tahun' not in str(first_row[0]).lower() else first_row
        data_rows = data_rows[1:]
    else:
        # Buat header default
        headers = ['Tahun'] + [f'Ukuran{i}' for i in range(1, len(first_row))]
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=headers)
    
    # Konversi tahun ke integer jika ada
    if 'Tahun' in df.columns:
        df['Tahun'] = df['Tahun'].astype(int)
    else:
        # Tambahkan kolom tahun jika tidak ada
        df.insert(0, 'Tahun', range(1, len(df) + 1))
    
    # Konversi kolom ukuran ke float
    for col in df.columns:
        if col != 'Tahun':
            try:
                df[col] = df[col].astype(float)
            except:
                pass
    
    return df

def extract_data_components(df):
    """Extract year column, years list, and size columns from DataFrame"""
    # Cari kolom tahun
    tahun_col = None
    for col in df.columns:
        if 'tahun' in col.lower():
            tahun_col = col
            break
        elif col.lower() == 'year':
            tahun_col = col
            break
    
    # Jika tidak ada kolom tahun, gunakan indeks
    if tahun_col is None and 'Tahun' in df.columns:
        tahun_col = 'Tahun'
    
    # Extract years
    if tahun_col:
        years = df[tahun_col].dropna().astype(int).tolist()
    else:
        years = list(range(1, len(df) + 1))
    
    # Extract size columns
    ukuran_cols = [c for c in df.columns if c != tahun_col]
    
    return tahun_col, years, ukuran_cols

def calculate_percentage_distribution(df, tahun_col, ukuran_cols):
    """Menghitung distribusi persentase per kategori per tahun"""
    distribution = {}
    
    for _, row in df.iterrows():
        tahun = int(row[tahun_col])
        total = sum([float(row[col]) for col in ukuran_cols])
        
        if total > 0:
            percentages = {}
            for col in ukuran_cols:
                percentage = (float(row[col]) / total) * 100
                percentages[col] = round(percentage, 2)
            
            # Pastikan total 100%
            total_percentage = sum(percentages.values())
            if abs(total_percentage - 100) > 0.1:
                # Normalisasi
                factor = 100 / total_percentage
                percentages = {k: round(v * factor, 2) for k, v in percentages.items()}
            
            distribution[tahun] = percentages
    
    return distribution

def calculate_yearly_calculations(data, forecast_values, alpha, years):
    """Menghitung semua perhitungan dari tahun pertama ke tahun berikutnya"""
    yearly_calculations = []
    
    # Tahun pertama: F1 = A1 (inisialisasi)
    yearly_calculations.append({
        'tahun': years[0] if years else "Tahun 1",
        'actual': round(data[0], 2),
        'forecast': round(data[0], 2),
        'calculation': f"F₁ = A₁ = {data[0]} (Inisialisasi)",
        'alpha': alpha,
        'type': 'inisialisasi'
    })
    
    # Tahun kedua dan seterusnya
    for t in range(1, len(data)):
        actual_prev = data[t-1]
        forecast_prev = forecast_values[t-1]
        forecast_current = forecast_values[t]
        
        calculation = f"F₍{t+1}₎ = {alpha} × {actual_prev} + {1-alpha} × {forecast_prev} = {forecast_current:.2f}"
        
        yearly_calculations.append({
            'tahun': years[t] if t < len(years) else f"Tahun {t+1}",
            'actual': round(data[t], 2),
            'forecast': round(forecast_current, 2),
            'calculation': calculation,
            'alpha': alpha,
            'type': 'smoothing',
            'actual_prev': round(actual_prev, 2),
            'forecast_prev': round(forecast_prev, 2)
        })
    
    return yearly_calculations

def calculate_detailed_metrics_by_year(actual_values, forecast_values, years):
    """Menghitung semua metrik evaluasi per tahun dengan detail lengkap"""
    yearly_metrics = []
    
    # Mulai dari tahun ke-2 karena tahun pertama hanya inisialisasi
    for i in range(1, len(actual_values)):
        actual = actual_values[i]
        forecast = forecast_values[i]
        
        absolute_error = abs(actual - forecast)
        squared_error = (actual - forecast) ** 2
        percentage_error = (absolute_error / actual) * 100 if actual != 0 else 0
        
        yearly_metrics.append({
            'tahun': years[i] if i < len(years) else f"Tahun {i+1}",
            'actual': round(actual, 2),
            'forecast': round(forecast, 2),
            'absolute_error': round(absolute_error, 2),
            'squared_error': round(squared_error, 2),
            'percentage_error': round(percentage_error, 2)
        })
    
    return yearly_metrics

def calculate_summary_metrics(yearly_metrics):
    """Menghitung summary metrics dari data tahunan"""
    if not yearly_metrics:
        return {}
    
    absolute_errors = [m['absolute_error'] for m in yearly_metrics]
    squared_errors = [m['squared_error'] for m in yearly_metrics]
    percentage_errors = [m['percentage_error'] for m in yearly_metrics]
    
    return {
        'mape': round(sum(percentage_errors) / len(percentage_errors), 2),
        'mae': round(sum(absolute_errors) / len(absolute_errors), 2),
        'mse': round(sum(squared_errors) / len(squared_errors), 2),
        'total_years': len(yearly_metrics)
    }

def generate_all_alpha_combinations():
    """Generate alpha dari 0.1 sampai 0.9 dengan step 0.1"""
    return [round(i/10, 1) for i in range(1, 10)]  # [0.1, 0.2, ..., 0.9]

def calculate_forecast_with_alpha(data, alpha, years):
    """Calculate forecast and metrics for a specific alpha dengan detail tahunan lengkap"""
    forecast_values = single_exponential_smoothing(data, alpha)
    
    if len(data) > 1 and len(forecast_values) > 1:
        # Hitung metrics per tahun
        yearly_metrics = calculate_detailed_metrics_by_year(data, forecast_values, years)
        summary_metrics = calculate_summary_metrics(yearly_metrics)
        
        # Hitung forecast untuk periode berikutnya
        forecast_next = calculate_forecast_next(data, alpha, forecast_values)
        
        # Hitung perhitungan tahunan
        yearly_calculations = calculate_yearly_calculations(data, forecast_values, alpha, years)
        
        return {
            'alpha': alpha,
            'mape': summary_metrics['mape'],
            'mae': summary_metrics['mae'],
            'mse': summary_metrics['mse'],
            'forecast_values': [round(x, 2) for x in forecast_values],
            'actual_values': data,
            'forecast_next': round(forecast_next, 2),
            'yearly_calculations': yearly_calculations,
            'yearly_metrics': yearly_metrics,
            'summary_metrics': summary_metrics
        }
    
    return None

def initialize_best_results():
    """Initialize best results tracking"""
    return {
        'alpha_mape': None,
        'mape': float('inf'),
        'forecast_mape': None,
        'alpha_mae': None,
        'mae': float('inf'),
        'alpha_mse': None,
        'mse': float('inf'),
        'forecast_next_mape': None,
        'forecast_next_mae': None,
        'forecast_next_mse': None
    }

def update_best_results(best_results, calculation, alpha):
    """Update best results dan simpan semua alpha yang memberikan nilai terbaik"""
    # Inisialisasi list jika belum ada
    if 'all_alpha_mape' not in best_results:
        best_results['all_alpha_mape'] = []
        best_results['all_alpha_mae'] = []
        best_results['all_alpha_mse'] = []
    
    # Untuk MAPE
    if calculation['mape'] < best_results['mape']:
        best_results['alpha_mape'] = alpha
        best_results['mape'] = calculation['mape']
        best_results['forecast_mape'] = calculation['forecast_values']
        best_results['forecast_next_mape'] = calculation['forecast_next']
        best_results['all_alpha_mape'] = [alpha]  # Reset list
    elif abs(calculation['mape'] - best_results['mape']) < 0.0001:
        # Jika sama, tambahkan ke list
        if alpha not in best_results['all_alpha_mape']:
            best_results['all_alpha_mape'].append(alpha)
    
    # Untuk MAE
    if calculation['mae'] < best_results['mae']:
        best_results['alpha_mae'] = alpha
        best_results['mae'] = calculation['mae']
        best_results['forecast_next_mae'] = calculation['forecast_next']
        best_results['all_alpha_mae'] = [alpha]
    elif abs(calculation['mae'] - best_results['mae']) < 0.0001:
        if alpha not in best_results['all_alpha_mae']:
            best_results['all_alpha_mae'].append(alpha)
    
    # Untuk MSE
    if calculation['mse'] < best_results['mse']:
        best_results['alpha_mse'] = alpha
        best_results['mse'] = calculation['mse']
        best_results['forecast_next_mse'] = calculation['forecast_next']
        best_results['all_alpha_mse'] = [alpha]
    elif abs(calculation['mse'] - best_results['mse']) < 0.0001:
        if alpha not in best_results['all_alpha_mse']:
            best_results['all_alpha_mse'].append(alpha)

def analyze_forecast_trend(forecast_next, actual_last):
    """Analyze forecast trend compared to last actual value"""
    if forecast_next > actual_last:
        return "Naik"
    elif forecast_next < actual_last:
        return "Turun"
    else:
        return "Stabil"

def create_final_result(ukuran_name, data, years, best_results, calculations, alpha_step=0.1):
    """Create final result dictionary for a size dengan data tahunan lengkap"""
    tahun_last = years[-1]
    actual_last = data[-1]
    forecast_last = best_results['forecast_mape'][-1] if best_results['forecast_mape'] else 0
    next_year = tahun_last + 1
    
    # Dapatkan calculation terbaik untuk MAPE
    best_mape_calc = next((calc for calc in calculations if calc['alpha'] == best_results['alpha_mape']), None)
    
    # Analyze trends
    data_trend = analyze_trend(data)
    forecast_trend = analyze_forecast_trend(best_results['forecast_next_mape'], actual_last)
    
    # Hitung metrics dengan presisi yang sama
    if best_mape_calc:
        # Gunakan metrics dari calculation langsung (jangan hitung ulang)
        mape = round(best_mape_calc['mape'], 4)
        mae = round(best_mape_calc['mae'], 4)
        mse = round(best_mape_calc['mse'], 4)
    else:
        # Fallback ke best_results
        mape = round(best_results['mape'], 4)
        mae = round(best_results['mae'], 4)
        mse = round(best_results['mse'], 4)
    
    result = {
        'ukuran': ukuran_name,
        'alpha': round(best_results['alpha_mape'], 4),
        'alpha_mape': round(best_results['alpha_mape'], 4),
        'alpha_mae': round(best_results['alpha_mae'], 4),
        'alpha_mse': round(best_results['alpha_mse'], 4),
        'mape': mape,  # Gunakan nilai yang sudah dihitung
        'mae': mae,    # Gunakan nilai yang sudah dihitung
        'mse': mse,    # Gunakan nilai yang sudah dihitung
        'tahun_last': tahun_last,
        'actual_last': round(actual_last, 2),
        'forecast_last': round(forecast_last, 2),
        'tahun_next': next_year,
        'forecast_next': round(best_results['forecast_next_mape'], 2),
        'forecast_next_mae': round(best_results['forecast_next_mae'], 2),
        'forecast_next_mse': round(best_results['forecast_next_mse'], 2),
        'data_trend': data_trend,
        'forecast_trend': forecast_trend,
        'data_points': len(data),
        'data_range': f"{min(data)} - {max(data)}",
        'data_variance': round(np.var(data), 4) if len(data) > 1 else 0
    }
    
    # Tambahkan data tahunan lengkap jika tersedia
    if best_mape_calc:
        result['yearly_calculations'] = best_mape_calc.get('yearly_calculations', [])
        result['yearly_metrics'] = best_mape_calc.get('yearly_metrics', [])
        result['metrics_detail'] = best_mape_calc.get('metrics_detail', {})
        
        # PASTIKAN: metrics di all_alpha_data sama dengan di result
        all_alpha_data = []
        for calc in calculations:
            # Gunakan metrics yang sama dari calculation
            all_alpha_data.append({
                'alpha': calc['alpha'],
                'mape': calc['mape'],  # Sama dengan yang di calculation
                'mae': calc['mae'],    # Sama dengan yang di calculation
                'mse': calc['mse'],    # Sama dengan yang di calculation
                'forecast_next': calc['forecast_next']
            })
        result['all_alpha_data'] = all_alpha_data
    
    chart_data = best_results['forecast_mape'] + [round(best_results['forecast_next_mape'], 2)] if best_results['forecast_mape'] else []
    
    return {
        'result': result,
        'chart_data': chart_data,
        'calculations': calculations
    }


def process_forecasting(ukuran_cols, df, tahun_col, years, alpha_step=0.1):
    """Process forecasting for all size columns"""
    results = []
    chart_data = {}
    all_calculations = {}
    percentage_distribution = {}
    
    # Hitung distribusi persentase
    percentage_distribution = calculate_percentage_distribution(df, tahun_col, ukuran_cols)
    
    # Batasi jumlah ukuran untuk menghindari overload
    max_sizes = min(len(ukuran_cols), 15)
    
    for idx, col in enumerate(ukuran_cols[:max_sizes]):
        try:
            data = df[col].dropna().astype(float).tolist()
            if len(data) < 2:
                continue
                
            result = process_single_size_forecast(col, data, years, alpha_step)
            if result:
                results.append(result['result'])
                chart_data[col] = result['chart_data']
                all_calculations[col] = result['calculations']
                
        except (ValueError, TypeError) as e:
            print(f"Error processing {col}: {str(e)}")
            continue
    
    return results, chart_data, all_calculations, percentage_distribution

def process_single_size_forecast(ukuran_name, data, years, alpha_step=0.1):
    """Process forecasting for a single size dengan years"""
    calculations = []
    best_results = initialize_best_results()
    
    # Tentukan alphas berdasarkan step
    if alpha_step == 0.1:  # 0.1 sampai 0.9
        alphas = [i / 10 for i in range(1, 10)]
    else:  # 0.01 sampai 0.99
        alphas = [i / 100 for i in range(1, 100)]
    
    for alpha in alphas:
        calculation = calculate_forecast_with_alpha(data, alpha, years)
        if calculation:
            calculations.append(calculation)
            update_best_results(best_results, calculation, alpha)
    
    # Dapatkan semua data alpha untuk modal detail
    all_alpha_data = []
    for calc in calculations:
        all_alpha_data.append({
            'alpha': calc['alpha'],
            'mape': calc['mape'],
            'mae': calc['mae'],
            'mse': calc['mse'],
            'forecast_next': calc['forecast_next']
        })
    
    # Create final result if valid forecasts exist
    if best_results['forecast_mape']:
        result = create_final_result(
            ukuran_name, data, years, best_results, calculations
        )
        # Tambahkan all_alpha_data ke result
        result['all_alpha_data'] = all_alpha_data
        result['alpha_step'] = alpha_step  # Tambahkan info step
        return result
    
    return None

def get_all_alpha_calculations_for_ukuran(ukuran_name, data, years, alpha_step=0.1):
    """Mendapatkan semua perhitungan untuk semua alpha"""
    all_alpha_data = []
    
    # Tentukan alphas berdasarkan step
    if alpha_step == 0.1:  # 0.1 sampai 0.9
        alphas = [i / 10 for i in range(1, 10)]
    else:  # 0.01 sampai 0.99
        alphas = [i / 100 for i in range(1, 100)]
    
    for alpha in alphas:
        calculation = calculate_forecast_with_alpha(data, alpha, years)
        if calculation:
            all_alpha_data.append({
                'alpha': alpha,
                'mape': calculation['mape'],
                'mae': calculation['mae'],
                'mse': calculation['mse'],
                'forecast_next': calculation['forecast_next'],
                'forecast_values': calculation['forecast_values'],
                'yearly_calculations': calculation.get('yearly_calculations', []),
                'yearly_metrics': calculation.get('yearly_metrics', [])
            })
    
    return all_alpha_data

import json

def prepare_context(results, chart_data, years, all_calculations, percentage_distribution):
    """Prepare context for template"""
    best_mape_all = min([r['mape'] for r in results]) if results else None
    total_calculations = sum(len(calcs) for calcs in all_calculations.values())
    
    # Prepare data for MAE and MSE best tables
    mae_best_results = []
    mse_best_results = []
    
    # Data untuk grafik distribusi persentase
    dist_years = list(percentage_distribution.keys())
    dist_data = {ukuran: [] for ukuran in all_calculations.keys()}
    
    for tahun, percentages in percentage_distribution.items():
        for ukuran, percentage in percentages.items():
            if ukuran in dist_data:
                dist_data[ukuran].append(percentage)
    
    # Ambil data alpha terbaik dari hasil yang sudah dihitung (results)
    alpha_terbaik_map = {}
    for result in results:
        ukuran = result['ukuran']
        alpha_terbaik_map[ukuran] = {
            'alpha_mape': result['alpha_mape'],
            'alpha_mae': result['alpha_mae'],
            'alpha_mse': result['alpha_mse'],
            'mape': result['mape'],
            'mae': result['mae'],
            'mse': result['mse'],
            'forecast_next_mape': result['forecast_next'],
            'forecast_next_mae': result['forecast_next_mae'],
            'forecast_next_mse': result['forecast_next_mse']
        }
    
    # Prepare mae_best_results dan mse_best_results menggunakan data dari results
    for result in results:
        ukuran = result['ukuran']
        
        if ukuran in all_calculations:
            calculations = all_calculations[ukuran]
            terbaik = alpha_terbaik_map.get(ukuran, {})
            
            # Buat MAE best result
            if 'alpha_mae' in terbaik:
                # Cari calculation untuk alpha MAE terbaik
                best_mae_calc = next((calc for calc in calculations 
                                    if abs(calc['alpha'] - terbaik['alpha_mae']) < 0.0001), None)
                
                if best_mae_calc:
                    mae_all_alpha_data = []
                    for calc in calculations:
                        mae_all_alpha_data.append({
                            'alpha': calc['alpha'],
                            'mape': calc['mape'],
                            'mae': calc['mae'],
                            'mse': calc['mse'],
                            'forecast_next': calc['forecast_next']
                        })
                    
                    mae_best_results.append({
                        'ukuran': ukuran,
                        'alpha': terbaik['alpha_mae'],
                        'alpha_mae': terbaik['alpha_mae'],
                        'mape': terbaik['mape'],
                        'mae': terbaik['mae'],
                        'mse': terbaik['mse'],
                        'forecast_next': terbaik['forecast_next_mae'],
                        'forecast_last': best_mae_calc['forecast_values'][-1] if best_mae_calc['forecast_values'] else result['actual_last'],
                        'tahun_last': result['tahun_last'],
                        'actual_last': result['actual_last'],
                        'tahun_next': result['tahun_next'],
                        'forecast_trend': analyze_forecast_trend(terbaik['forecast_next_mae'], result['actual_last']),
                        'yearly_calculations': best_mae_calc.get('yearly_calculations', []),
                        'yearly_metrics': best_mae_calc.get('yearly_metrics', []),
                        'metrics_detail': best_mae_calc.get('metrics_detail', {}),
                        'all_alpha_data': mae_all_alpha_data,
                        # ✅ ADD JSON serialization
                        'all_alpha_data_json': json.dumps(mae_all_alpha_data),
                        'yearly_calculations_json': json.dumps(best_mae_calc.get('yearly_calculations', [])),
                        'yearly_metrics_json': json.dumps(best_mae_calc.get('yearly_metrics', []))
                    })
            
            # Buat MSE best result
            if 'alpha_mse' in terbaik:
                # Cari calculation untuk alpha MSE terbaik
                best_mse_calc = next((calc for calc in calculations 
                                    if abs(calc['alpha'] - terbaik['alpha_mse']) < 0.0001), None)
                
                if best_mse_calc:
                    mse_all_alpha_data = []
                    for calc in calculations:
                        mse_all_alpha_data.append({
                            'alpha': calc['alpha'],
                            'mape': calc['mape'],
                            'mae': calc['mae'],
                            'mse': calc['mse'],
                            'forecast_next': calc['forecast_next']
                        })
                    
                    mse_best_results.append({
                        'ukuran': ukuran,
                        'alpha': terbaik['alpha_mse'],
                        'alpha_mse': terbaik['alpha_mse'],
                        'mape': terbaik['mape'],
                        'mae': terbaik['mae'],
                        'mse': terbaik['mse'],
                        'forecast_next': terbaik['forecast_next_mse'],
                        'forecast_last': best_mse_calc['forecast_values'][-1] if best_mse_calc['forecast_values'] else result['actual_last'],
                        'tahun_last': result['tahun_last'],
                        'actual_last': result['actual_last'],
                        'tahun_next': result['tahun_next'],
                        'forecast_trend': analyze_forecast_trend(terbaik['forecast_next_mse'], result['actual_last']),
                        'yearly_calculations': best_mse_calc.get('yearly_calculations', []),
                        'yearly_metrics': best_mse_calc.get('yearly_metrics', []),
                        'metrics_detail': best_mse_calc.get('metrics_detail', {}),
                        'all_alpha_data': mse_all_alpha_data,
                        # ✅ ADD JSON serialization
                        'all_alpha_data_json': json.dumps(mse_all_alpha_data),
                        'yearly_calculations_json': json.dumps(best_mse_calc.get('yearly_calculations', [])),
                        'yearly_metrics_json': json.dumps(best_mse_calc.get('yearly_metrics', []))
                    })
    
    # Update all_calculations dengan informasi "terbaik" dari results
    for ukuran, calculations in all_calculations.items():
        if ukuran in alpha_terbaik_map:
            terbaik = alpha_terbaik_map[ukuran]
            
            # Tandai alpha terbaik di setiap calculation
            for calc in calculations:
                calc['is_best_mape'] = abs(calc['alpha'] - terbaik['alpha_mape']) < 0.0001
                calc['is_best_mae'] = abs(calc['alpha'] - terbaik['alpha_mae']) < 0.0001
                calc['is_best_mse'] = abs(calc['alpha'] - terbaik['alpha_mse']) < 0.0001
    
    # ✅ CRITICAL FIX: Update results dengan JSON serialization
    for result in results:
        ukuran = result['ukuran']
        if ukuran in all_calculations:
            calculations = all_calculations[ukuran]
            
            # Pastikan kita menggunakan calculation untuk alpha MAPE terbaik
            terbaik = alpha_terbaik_map.get(ukuran, {})
            best_mape_calc = next((calc for calc in calculations 
                                 if abs(calc['alpha'] - terbaik.get('alpha_mape', 0)) < 0.0001), None)
            
            if best_mape_calc:
                result['actual_values'] = best_mape_calc['actual_values']
                result['forecast_values'] = best_mape_calc['forecast_values']
                result['yearly_calculations'] = best_mape_calc.get('yearly_calculations', [])
                result['yearly_metrics'] = best_mape_calc.get('yearly_metrics', [])
                result['summary_metrics'] = best_mape_calc.get('summary_metrics', {})
                
                # Buat all_alpha_data dengan flag terbaik
                all_alpha_data = []
                for calc in calculations:
                    all_alpha_data.append({
                        'alpha': calc['alpha'],
                        'mape': calc['mape'],
                        'mae': calc['mae'],
                        'mse': calc['mse'],
                        'forecast_next': calc['forecast_next'],
                        'is_best_mape': calc.get('is_best_mape', False),
                        'is_best_mae': calc.get('is_best_mae', False),
                        'is_best_mse': calc.get('is_best_mse', False)
                    })
                
                result['all_alpha_data'] = all_alpha_data
                
                # ✅ SERIALIZE TO JSON for template
                result['all_alpha_data_json'] = json.dumps(all_alpha_data)
                result['yearly_calculations_json'] = json.dumps(result['yearly_calculations'])
                result['yearly_metrics_json'] = json.dumps(result['yearly_metrics'])
    
    return {
        'results': results,
        'chart_data': chart_data,
        'years': years,
        'best_mape': best_mape_all,
        'all_calculations': all_calculations,
        'total_calculations': total_calculations,
        'mae_best_results': mae_best_results,
        'mse_best_results': mse_best_results,
        'percentage_distribution': percentage_distribution,
        'dist_years': dist_years,
        'dist_data': dist_data
    }
    
def forecast(request):
    if request.method == 'POST':
        try:
            # Get alpha step dari form dan pastikan float
            alpha_step_input = request.POST.get('alpha_step', '0.1')
            
            # Konversi ke float, jika ada masalah gunakan default
            try:
                alpha_step = float(alpha_step_input)
            except (ValueError, TypeError):
                alpha_step = 0.1  # Default
            
            # Pastikan hanya 0.1 atau 0.01
            if alpha_step not in [0.1, 0.01]:
                alpha_step = 0.1
            
            print(f"DEBUG: alpha_step = {alpha_step} (type: {type(alpha_step)})")
            
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

            # Process forecasting for all sizes dengan alpha_step yang dipilih
            results, chart_data, all_calculations, percentage_distribution = process_forecasting(
                ukuran_cols, df, tahun_col, years, alpha_step
            )
            
            if not results:
                return render(request, 'forecasting/result.html', {
                    'error': 'Tidak ada data yang dapat diproses untuk forecasting.'
                })
            
            # Prepare context dengan info alpha_step
            context = prepare_context(
                results, chart_data, years, all_calculations, percentage_distribution
            )
            context['alpha_step'] = alpha_step
            context['alpha_count'] = 99 if alpha_step == 0.01 else 9
            
            # DEBUG: Tampilkan di console
            print(f"DEBUG: Sending to template: alpha_step={alpha_step}, type={type(alpha_step)}")
            
            return render(request, 'forecasting/result.html', context)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return render(request, 'forecasting/result.html', {
                'error': f'Terjadi kesalahan: {str(e)}'
            })

    return render(request, 'forecasting/index.html')
@csrf_exempt
def api_forecast(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Gunakan metode POST.'}, status=405)

    try:
        df = process_input_data(request)
        if df is None:
            return JsonResponse({'error': 'Harap kirim file CSV/Excel atau JSON valid.'}, status=400)

        tahun_col, years, ukuran_cols = extract_data_components(df)
        if not ukuran_cols:
            return JsonResponse({'error': 'Tidak ditemukan kolom ukuran.'}, status=400)

        results, chart_data, all_calculations, percentage_distribution = process_forecasting(
            ukuran_cols, df, tahun_col, years
        )
        
        best_mape_all = min([r['mape'] for r in results]) if results else None
        
        return JsonResponse({
            'status': 'success',
            'results': results,
            'best_mape': best_mape_all,
            'chart_data': chart_data,
            'years': years,
            'all_calculations': all_calculations,
            'percentage_distribution': percentage_distribution,
            'total_calculations': sum(len(calcs) for calcs in all_calculations.values())
        }, status=200, json_dumps_params={'ensure_ascii': False})

    except Exception as e:
        return JsonResponse({'error': f'Gagal memproses data: {str(e)}'}, status=400)