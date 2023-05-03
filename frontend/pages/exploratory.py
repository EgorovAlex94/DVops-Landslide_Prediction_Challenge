import streamlit as st
import yaml
from src.data.get_data import load_data, get_dataset
from src.plotting.charts import barplot_group, heatmap, norm_target


CONFIG_PATH = "../config/params.yml"

st.markdown("# Exploratory data analysis")
st.sidebar.markdown("# Exploratory data analysis")

with open(CONFIG_PATH) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_dataset(dataset_path=config["preprocessing"]["train_path"])
    st.write(data.head())

# plotting with checkbox
balance_response = st.sidebar.checkbox("Проверка баланса классов")
heatmap_response = st.sidebar.checkbox("Корреляции признак-таргет")
geology_response = st.sidebar.checkbox("Зависимость таргета от геологической местности")

if balance_response:
    st.pyplot(
        norm_target(
            data=data,
            col="Label",
            title="Баланс классов"
        )
    )

if heatmap_response:
    st.pyplot(
        heatmap(
            data=data,
            cols=['1_elevation', '1_aspect', '1_slope', '1_placurv',
                  '1_procurv', '1_lsfactor', '1_twi', '1_sdoif', 'Label'],
            title="Корреляции признак-таргет",
        )
    )

if geology_response:
    st.pyplot(
        barplot_group(
            data=data,
            col_main="1_geology",
            col_group="Label",
            title="Тагрет - тип геологической местности"
        )
    )