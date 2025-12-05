import scipy.stats as stats

def estimate_cost_interval(predicted_cost, mape, confidence=0.95):
    """Оценка доверительного интервала стоимости"""
    # Предполагаем нормальное распределение ошибок
    std_error = predicted_cost * mape / 100
    
    z_score = stats.norm.ppf(1 - (1 - confidence)/2)
    
    lower_bound = predicted_cost - z_score * std_error
    upper_bound = predicted_cost + z_score * std_error
    
    return lower_bound, upper_bound

# Пример
cost_range = estimate_cost_interval(100000, 15)
print(f"95% ДИ: [{cost_range[0]:.0f}, {cost_range[1]:.0f}] руб.")