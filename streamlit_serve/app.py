from typing import Literal

from traitlets import default


def main():
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    import streamlit as st
    import platform
    import pandas as pd
    import plotly.express as px


    def plot_feature_idx_frequency(df: pd.DataFrame):
        # Get the frequency counts for each feature_idx
        feature_counts = df['feature_idx'].value_counts().reset_index()
        feature_counts.columns = ['feature_idx', 'count']
        # Convert the counts to a numpy array
        feature_counts_array = feature_counts['count'].to_numpy()
        
        # Plot the histogram using Plotly
        fig = px.histogram(x=feature_counts_array, title='Feature Index Frequency')
        fig.update_layout(
            xaxis_title='Count',
            yaxis_title='Feature Indices with count',
            bargap=0.1,
            hovermode='x unified',
            autosize=False,
            modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
            margin=dict(t=50, l=50, b=50, r=50)
        )
        
        return fig
    
    def get_counted_df(df : pd.DataFrame) -> pd.DataFrame:
        feature_counts = df['feature_idx'].value_counts().reset_index()
        feature_counts.columns = ['feature_idx', 'count']
        return feature_counts

    def get_active_features_filtered(
        filtered_df : pd.DataFrame, 
        min_count : int, 
        method : Literal['leq', 'eq', 'geq']
    ) -> pd.DataFrame:
        if method == 'leq':
            return filtered_df[filtered_df['count'] <= min_count].head(10)
        elif method == 'eq':
            return filtered_df[filtered_df['count'] == min_count].head(10)
        elif method == 'geq':
            return filtered_df[filtered_df['count'] >= min_count].head(10)

    def get_active_features_stats(total_features : int, df : pd.DataFrame) -> tuple[int, float]:
        active_features = len(df['feature_idx'].unique())
        return active_features, active_features/total_features
        
    if platform.system() == 'Linux':
        from common import PATH
        models_path = os.path.join(PATH, 'laion2b_autoencoders')
    else:
        models_path = 'laion2b_autoencoders'


    models = os.listdir(models_path)

        # Select a feature index
    path = 'autoencoder_d_hidden_76800_dict_mult_100'
    default_index = models.index(path) if path in models else 0
    path = st.selectbox('Select Model', models, index=default_index)
    model = st.selectbox('select version', os.listdir(f'{models_path}/{path}'))
    path = f'{models_path}/{path}/{model}/activations/laion_acts_all.parquet'
    with st.spinner('Loading data...'):
        
        df = pd.read_parquet(path)
    st.success('Data loaded!')

    total_features = int(path.split('_')[4]) #TODO watch out for path changes
    active_features, active_features_ratio = get_active_features_stats(total_features, df)
    st.write(f"Total features: {total_features}, Active features: {active_features}, Active features ratio: {active_features_ratio}")
    st.plotly_chart(plot_feature_idx_frequency(df))

    filtered_df = get_counted_df(df)
    min_count = st.slider('Minimum count threshold for features', min_value=1, max_value=int(filtered_df['count'].max()), value=10)
    active_features = get_active_features_filtered(filtered_df, min_count, 'geq')
    st.write(f"Active features: {len(active_features)}, percentage: {len(active_features)/total_features}")

    st.write(f"Ten features with count {min_count}")
    features = get_active_features_filtered(filtered_df, min_count, 'eq')
    st.table(features)

    feature_index = st.selectbox('Select Feature Index', df['feature_idx'].unique())

    # Filter dataframe based on selected feature index
    filtered_data = df[df['feature_idx'] == feature_index].sort_values('quantized_activation', ascending=False)
    st.write(f"Feature Index: {feature_index}, samples: {len(filtered_data)}")

    # Display images and captions
    quantizations = reversed([i for i in range(10)])
    for quantization in quantizations:
        quant_df = filtered_data[filtered_data['quantized_activation'] == quantization]
        st.write(f"Quantization: {quantization}, samples: {len(quant_df)}")
        if len(quant_df) == 0:
            continue

        #ten_random_samples = quant_df.sample(min(5, len(quant_df)), replace=False)
        for idx, row in quant_df.iterrows():
            st.image(row['url'], caption=f"{row['caption']}, quantized_activation : {quantization} activation: {row['activation']}")

if __name__ == "__main__":
    main()