import json
import matplotlib.pyplot as plt
import numpy as np


# MHA baselines (B=16, H=16, D=128)
mi355x_mha_baselines_causal = {
    "triton": {
        "1024": 147,
        "2048": 179,
        "4096": 215,
        "8192": 237,
        "16384": 251,
    },
    "ck": {
        "1024": 517.46,
        "2048": 749.49,
        "4096": 911.59,
        "8192": 961.69,
        "16384": 1080.52,
    },
    "torch": {
        "1024": 109.51,
        "2048": 156.71,
        "4096": 142.82,
        "8192": 224.01,
        "16384": 259.14,
    },
}

mi350x_mha_baselines_causal = {
    "triton": {
        "1024": 139.080476,
        "2048": 163.599388,
        "4096": 188.696534,
        "8192": 200.207510,
        "16384": 206.924580,
    },
    "ck": {
        "1024": 489.78,
        "2048": 698.71,
        "4096": 798.36,
        "8192": 868.01,
        "16384": 927.37,
    },
    "torch": {
        "1024": 103.58,
        "2048": 145.20,
        "4096": 131.52,
        "8192": 208.90,
        "16384": 239.38,
    },
}

mi355x_mha_baselines_non_causal = { 
    # triton not available for non-causal bwd attn
    "ck": {
        "1024": 365.47,
        "2048": 427.98,
        "4096": 447.64,
        "8192": 469.74,
        "16384": 470.02,
    },
    "torch": {
        "1024": 220.27,
        "2048": 273.06,
        "4096": 301.24,
        "8192": 309.30,
        "16384": 311.66,
    },
}

mi350x_mha_baselines_non_causal = { 
    # triton not available for non-causal bwd attn
    "ck": {
        "1024": 344.08,
        "2048": 404.38,
        "4096": 419.94,
        "8192": 445.81,
        "16384": 446.31,
    },
    "torch": {
        "1024": 207.01,
        "2048": 254.13,
        "4096": 274.05,
        "8192": 286.45,
        "16384": 290.61,
    },
}

# GQA baselines (B=16, Q_HEADS=64, KV_HEADS=8, D=128)
mi355x_gqa_baselines_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 543.11,
        "2048": 810.94,
        "4096": 945.84,
        "8192": 977.65,
        "16384": 1079.00,
    },
    "torch": {
        "1024": 109.51,
        "2048": 156.71,
        "4096": 142.82,
        "8192": 224.01,
        "16384": 259.14,
    },
}

mi350x_gqa_baselines_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 510.38,
        "2048": 746.08,
        "4096": 819.23,
        "8192": 879.51,
        "16384": 911.87,
    },
    "torch": {
        "1024": 103.58,
        "2048": 145.20,
        "4096": 131.52,
        "8192": 208.90,
        "16384": 239.38,
    },
}

mi355x_gqa_baselines_non_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 368.51,
        "2048": 428.20,
        "4096": 445.61,
        "8192": 464.20,
        "16384": 467.18,
    },
    "torch": {
        "1024": 220.27,
        "2048": 273.06,
        "4096": 301.24,
        "8192": 309.30,
        "16384": 311.66,
    },
}

mi350x_gqa_baselines_non_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 348.40,
        "2048": 403.89,
        "4096": 418.79,
        "8192": 434.00,
        "16384": 439.18,
    },
    "torch": {
        "1024": 207.01,
        "2048": 254.13,
        "4096": 274.05,
        "8192": 286.45,
        "16384": 290.61,
    },
}


colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B"]


def process_data(data_list):
    """Separate numeric values and OOM indices"""
    values = []
    oom_indices = []
    for i, val in enumerate(data_list):
        if val == "OOM":
            values.append(0)  # Use 0 for bar height
            oom_indices.append(i)
        else:
            values.append(val)
    return values, oom_indices


for device in ['mi350x', 'mi355x']:
    for setting in ['mha_causal_bkwd', 'mha_non_causal_bkwd', 'gqa_causal_bkwd', 'gqa_non_causal_bkwd']:
        
        # Read data
        try:
            # Map setting to filename
            if setting == 'mha_causal_bkwd':
                filename = f'benchmark/{device}_mha_bkwd_causal.json'
            elif setting == 'mha_non_causal_bkwd':
                filename = f'benchmark/{device}_mha_bkwd_non_causal.json'
            elif setting == 'gqa_causal_bkwd':
                filename = f'benchmark/{device}_gqa_bkwd_causal.json'
            elif setting == 'gqa_non_causal_bkwd':
                filename = f'benchmark/{device}_gqa_bkwd_non_causal.json'
            
            with open(filename, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # Extract data for plotting
        matrix_sizes = sorted([int(size) for size in data.keys()])
        aiter_tflops = [data[str(size)]['tflops_ref'] for size in matrix_sizes]
        tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

        # Get baseline data based on setting
        triton_tflops = []
        torch_tflops = []
        ck_tflops = []
        
        if setting == 'mha_causal_bkwd' and device == 'mi355x':
            triton_tflops = [mi355x_mha_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_mha_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_non_causal_bkwd' and device == 'mi355x':
            # No triton for non-causal backward
            torch_tflops = [mi355x_mha_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_causal_bkwd' and device == 'mi355x':
            # No triton for GQA
            torch_tflops = [mi355x_gqa_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_non_causal_bkwd' and device == 'mi355x':
            torch_tflops = [mi355x_gqa_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_causal_bkwd' and device == 'mi350x':
            triton_tflops = [mi350x_mha_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi350x_mha_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_mha_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_non_causal_bkwd' and device == 'mi350x':
            torch_tflops = [mi350x_mha_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_mha_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_causal_bkwd' and device == 'mi350x':
            torch_tflops = [mi350x_gqa_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_gqa_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_non_causal_bkwd' and device == 'mi350x':
            torch_tflops = [mi350x_gqa_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_gqa_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]

        # Process data to separate OOM values
        triton_vals, triton_oom = process_data(triton_tflops) if triton_tflops else ([], [])
        torch_vals, torch_oom = process_data(torch_tflops) if torch_tflops else ([], [])
        ck_vals, ck_oom = process_data(ck_tflops) if ck_tflops else ([], [])

        # Calculate max for numeric values only
        numeric_vals = aiter_tflops + tk_tflops
        if triton_vals:
            numeric_vals.extend([v for v in triton_vals if v != 0])
        if torch_vals:
            numeric_vals.extend([v for v in torch_vals if v != 0])
        if ck_vals:
            numeric_vals.extend([v for v in ck_vals if v != 0])
        max_tflops = max(numeric_vals) if numeric_vals else 100

        # Create bar chart
        x = np.arange(len(matrix_sizes))
        width = 0.19

        fig, ax = plt.subplots(figsize=(10, 6))

        if triton_vals:
            # 5 bars: PyTorch, Triton, CK, AITER, HipKittens
            first_bar_start = x - 2*width
            second_bar_start = x - width
            third_bar_start = x
            fourth_bar_start = x + width
            fifth_bar_start = x + 2*width

            bars3 = ax.bar(first_bar_start, torch_vals, width, label='PyTorch SDPA', color=colors[1])
            bars2 = ax.bar(second_bar_start, triton_vals, width, label='Triton', color=colors[2])
            bars4 = ax.bar(third_bar_start, ck_vals, width, label='Composable Kernel', color=colors[4])
            bars0 = ax.bar(fourth_bar_start, aiter_tflops, width, label='AITER (ASM)', color=colors[0])
            bars1 = ax.bar(fifth_bar_start, tk_tflops, width, label='HipKittens', color=colors[3])
        else:
            # 4 bars: PyTorch, CK, AITER, HipKittens
            first_bar_start = x - 1.5*width
            second_bar_start = x - 0.5*width
            third_bar_start = x + 0.5*width
            fourth_bar_start = x + 1.5*width

            bars3 = ax.bar(first_bar_start, torch_vals, width, label='PyTorch SDPA', color=colors[1])
            bars4 = ax.bar(second_bar_start, ck_vals, width, label='Composable Kernel', color=colors[4])
            bars0 = ax.bar(third_bar_start, aiter_tflops, width, label='AITER', color=colors[0])
            bars1 = ax.bar(fourth_bar_start, tk_tflops, width, label='HipKittens', color=colors[3])

        fontsize = 11

        # Plot X markers for OOM
        oom_height = 50  # Position X near top of chart
        if torch_oom:
            for idx in torch_oom:
                offset = -2*width if triton_vals else -1.5*width
                ax.plot(x[idx] + offset, oom_height, 'x', color=colors[1],
                       markersize=15, markeredgewidth=3)
                ax.text(x[idx] + offset, oom_height + max_tflops * 0.03,
                       'OOM', ha='center', va='bottom', fontsize=fontsize, color=colors[1])

        # Add value labels on bars
        for bar, value in zip(bars0, aiter_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        for bar, value in zip(bars1, tk_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        if triton_vals:
            for bar, value in zip(bars2, triton_vals):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        if torch_vals:
            for i, (bar, value) in enumerate(zip(bars3, torch_vals)):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        if ck_vals:
            for bar, value in zip(bars4, ck_vals):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        # Parse setting name for title
        setting_parts = setting.split('_')
        attn_type = setting_parts[0].upper()  # MHA or GQA
        causal_mode = 'Causal' if 'causal' in setting and 'non_causal' not in setting else 'Non-Causal'

        # add some padding to the top of the y-axis to prevent label overlap
        ax.set_ylim(0, max_tflops * 1.15)
        ax.set_xlabel('Sequence Length', fontsize=16)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
        ax.set_title(f'{attn_type} {causal_mode} Backward Performance Comparison {device.upper()}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(matrix_sizes, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=14)

        plt.tight_layout()
        plt.show()

        output_file = f'{device}_{setting}_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

        # Print summary
        print(f"Sequence lengths tested: {matrix_sizes}")
        print(f"AITER TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
        print(f"HipKittens TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")

