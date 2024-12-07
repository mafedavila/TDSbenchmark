import json
import matplotlib.pyplot as plt

def plot_metrics(json_file):
    with open(json_file, 'r') as f:
        metrics = json.load(f)
    
    columns = list(metrics['Jensen Shannon Divergence'].keys())
    
    # Plotting Jensen Shannon Divergence
    plt.figure(figsize=(10, 5))
    plt.bar(columns, metrics['Jensen Shannon Divergence'].values())
    plt.xlabel('Columns')
    plt.ylabel('Jensen Shannon Divergence')
    plt.title('Jensen Shannon Divergence per Column')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Plotting Wasserstein Distance
    plt.figure(figsize=(10, 5))
    plt.bar(columns, metrics['Wasserstein Distance'].values())
    plt.xlabel('Columns')
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance per Column')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Plotting Kolmogorov Smirnov Test (KS Statistic)
    ks_stats = [metrics['Kolmogorov Smirnov Test'][col]['ks_stat'] for col in columns]
    plt.figure(figsize=(10, 5))
    plt.bar(columns, ks_stats)
    plt.xlabel('Columns')
    plt.ylabel('KS Statistic')
    plt.title('Kolmogorov Smirnov Test (KS Statistic) per Column')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Plotting Kolmogorov Smirnov Test (KS p-value)
    ks_p_values = [metrics['Kolmogorov Smirnov Test'][col]['ks_p_value'] for col in columns]
    plt.figure(figsize=(10, 5))
    plt.bar(columns, ks_p_values)
    plt.xlabel('Columns')
    plt.ylabel('KS p-value')
    plt.title('Kolmogorov Smirnov Test (KS p-value) per Column')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_metrics('ctabganplus_adult_statistics_evaluation.json')
