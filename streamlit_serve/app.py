
def main():
    import streamlit as st
    import os
    import platform
    import pandas as pd
    
    if platform.system() == 'Linux':
        from common import PATH
        models_path = os.path.join(PATH, 'laion2b_autoencoders')
    else:
        models_path = 'laion2b_autoencoders'
   

    models = os.listdir(models_path)

    # Select a feature index
    path = 'autoencoder_d_hidden_6144_lr_5e-05_dict_mult_8_epoch_1'
    path = st.selectbox('Select Model', models)
    path = f'{models_path}/{path}/activations/laion_acts_all.parquet'
    with st.spinner('Loading data...'):
        
        df = pd.read_parquet(path)
    st.success('Data loaded!')

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

        ten_random_samples = quant_df.sample(min(5, len(quant_df)), replace=False)
        for idx, row in ten_random_samples.iterrows():
            st.image(row['url'], caption=f"{row['caption']}, quantized_activation : {quantization} activation: {row['activation']}")

if __name__ == "__main__":
    main()