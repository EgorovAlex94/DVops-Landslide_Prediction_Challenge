import streamlit as st
import yaml
from src.train.training import start_training

CONFIG_PATH = "../config/params.yml"


st.markdown("# Train page")
st.sidebar.markdown("# Train page")

st.markdown("# Training model LightGBM")
# get params
with open(CONFIG_PATH) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# enpoint
endpoint = config["endpoints"]["train"]

if st.button("Start training"):
    start_training(config=config, endpoint=endpoint)