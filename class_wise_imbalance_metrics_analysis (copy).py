import os, ast, re, argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def parse_params(s):
    if not isinstance(s, str) or ":" not in s:
        return {}
    
    s = s.strip().strip("{}").replace('\n', '')
    parts = re.split(r',(?![^\[]*\])', s)
    
    params = {}
    for part in parts:
        if ":" not in part:
            continue
            
        k, v = part.split(":", 1)
        k = k.strip().strip("'").strip('"')
        v = v.strip().strip("'").strip('"')
        
        try:
            if "." in v:
                params[k] = float(v)
            else:
                params[k] = int(v)
        except ValueError:
            # --- NEW LOGIC START ---
            # If the value contains '::', take only the part before it
            if "::" in v:
                v = v.split("::")[0]
            # --- NEW LOGIC END ---
            params[k] = v
            
    return params

def generate_final_clinical_dashboard(root_path):
    # 1. Extract Study Name and Setup Directory
    study_name = os.path.basename(root_path.rstrip('/'))
    output_dir = f"class_wise_{study_name}_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Gather Data
    summary_df = pd.read_csv(os.path.join(root_path, 'trial_metrics.csv'))
    all_rows = []
    
    # Calculate global total for precise imbalance ratios
    first_trial_id = summary_df['trial'].iloc[0]
    test_first = pd.read_csv(os.path.join(root_path, f"trial_{first_trial_id}", 'test_final.csv'))
    total_samples = sum([sum(row) for row in ast.literal_eval(test_first['confusion_matrix'].iloc[0])])

    for _, tr in summary_df.iterrows():
        t_file = os.path.join(root_path, f"trial_{tr['trial']}", 'test_final.csv')
        if not os.path.exists(t_file): continue
        params = parse_params(tr['trial_params'])
        test_df = pd.read_csv(t_file)
        cls, f1 = ast.literal_eval(test_df['class_names'].iloc[0]), ast.literal_eval(test_df['classwise_f1'].iloc[0])
        supp = [sum(r) for r in ast.literal_eval(test_df['confusion_matrix'].iloc[0])]
        for i in range(len(cls)):
            all_rows.append({'Trial': tr['trial'], 'Language': cls[i], 'F1': f1[i], 'Support': supp[i], **params})
    
    df = pd.DataFrame(all_rows)
    hparams = [c for c in df.columns if c not in ['Trial', 'Language', 'F1', 'Support', 'Magnitude']]
    colors = {'1_Extreme (<10k)': '#e41a1c', '2_High (10k-40k)': '#ff7f00', '3_Medium (40k-80k)': '#4daf4a', '4_Low (>80k)': '#377eb8'}

    for hp in hparams:
        is_lr = "lr" in hp.lower() or "learning_rate" in hp.lower()
        num_langs = len(df['Language'].unique())
        dynamic_height = max(800, num_langs * 30)

        # 3. Create Multi-Panel Grid
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=(f"Scatter: {hp}", f"Surgical Heatmap: {hp}"),
            column_widths=[0.35, 0.65], 
            horizontal_spacing=0.12 
        )

        # SCATTER (Box 1)
        df['Magnitude'] = df['Support'].apply(lambda n: '1_Extreme (<10k)' if n <= 10000 else ('2_High (10k-40k)' if n <= 40000 else ('3_Medium (40k-80k)' if n <= 80000 else '4_Low (>80k)')))
        for mag, color in colors.items():
            m_df = df[df['Magnitude'] == mag]
            if m_df.empty: continue
            fig.add_trace(
                go.Scatter(
                    x=m_df[hp], y=m_df['F1'], mode='markers', name=mag,
                    marker=dict(color=color, size=11, line=dict(width=0.8, color='white')),
                    text=m_df['Language'],
                    hovertemplate="Lang: %{text}<br>F1: %{y:.4f}<br>Val: %{x}"
                ), row=1, col=1
            )

        # HEATMAP (Box 2)
        pivot = df.pivot_table(index='Language', columns=hp, values='F1', aggfunc='mean')
        lang_stats = df.groupby('Language')['Support'].first().sort_values(ascending=True)
        pivot = pivot.reindex(lang_stats.index)
        
        # Strictly existing values for X-axis
        heatmap_x = [f"{float(c):.1e}" if is_lr else str(c) for c in pivot.columns]
        # Linear labels: Name -> N -> Ratio
        heatmap_y = [f"{l} | N={lang_stats[l]} | Ratio={lang_stats[l]/total_samples:.4f}" for l in pivot.index]

        fig.add_trace(
            go.Heatmap(
                z=pivot.values, x=heatmap_x, y=heatmap_y,
                colorscale="RdYlGn", 
                colorbar=dict(title="F1 Score", x=1.18, thickness=20), # Pushed past labels
                xgap=3, ygap=3, # Surgical cell separation
                hovertemplate="Lang: %{y}<br>Val: %{x}<br>F1: %{z:.4f}"
            ), row=1, col=2
        )

        # 4. REARRANGEMENT & GRID REFINEMENT
        # Labels to the right to prevent muddle
        fig.update_yaxes(side="right", showgrid=True, gridcolor='black', gridwidth=0.5, griddash='solid', row=1, col=2)
        fig.update_xaxes(showgrid=True, gridcolor='black', gridwidth=0.5, griddash='solid', tickangle=45, row=1, col=2)
        
        if is_lr:
            fig.update_xaxes(type="log", tickformat=".1e", row=1, col=1)

        # Thick Divider
        fig.add_shape(type="line", x0=0.37, x1=0.37, y0=0, y1=1, xref="paper", yref="paper", line=dict(color="Black", width=3))

        fig.update_layout(
            title_text=f"Study: {study_name} | Parameter: {hp}",
            height=dynamic_height, width=2100, template="plotly_white",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=-0.12),
            margin=dict(l=150, r=550, t=100, b=100) # Maximum right-side clearance
        )
        
        fig.write_html(os.path.join(output_dir, f"dual_proof_{hp}.html"))

    print(f"Study '{study_name}' analysis complete. Results in: {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_dir', type=str, required=True)
    generate_final_clinical_dashboard(parser.parse_args().metrics_dir)
