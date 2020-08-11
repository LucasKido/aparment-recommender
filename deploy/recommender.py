import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
from time import sleep, time
from typing import Optional
from datetime import timedelta, datetime, date, time
from prep_utils import *


def select_block_container_style():
	st.markdown(
		f"""
<style>
	.reportview-container .main .block-container{{
		max-width: 1200px;
		padding-top: 5rem;
		padding-right: 3rem;
		padding-left: 2rem;
		padding-bottom: 10rem;
	}}
</style>
""",
		unsafe_allow_html=True,
	)

def set_variables():
	lgbm = jb.load('./lgbm_20200720.pkl.z')
	rf = jb.load('random_forest_20200720.pkl.z')
	enc = jb.load('onehotvec_20200720.pkl.z')
	return lgbm, rf, enc

def encode_variables(enc, df):
	cat_cols = ['bairro', 'estacao']
	tmp = df.drop(['id', 'endereco', 'update_time', 'apartment_link'], axis=1)
	cat_feat = enc.transform(df[cat_cols])
	cat_feat = pd.DataFrame(cat_feat, index=df.index)
	cat_feat = pd.merge(tmp.drop(cat_cols, axis=1), cat_feat, left_index=True, right_index=True)
	return cat_feat

def get_predictions(dataset, rf, lgbm):
	ypred_rf = rf.predict_proba(dataset)[:,1]
	ypred_lgbm = lgbm.predict_proba(dataset)[:,1]
	ypred_ens = 0.5*ypred_rf + 0.5*ypred_lgbm
	return ypred_ens

def format_data(df, dataset, rf, lgbm):
	ypred = get_predictions(dataset, rf, lgbm)
	data = pd.merge(df, pd.DataFrame(ypred, columns=['score']), left_index=True, right_index=True)
	data = data.sort_values('score', ascending=False).reset_index(drop=True)
	return data

def set_sidebar():
	st.sidebar.title('Settings')
	no_display = st.sidebar.number_input('Quantidade de apartamentos', min_value=5, max_value=20, value=10, step=1)
	st.sidebar.header('Options')
	mapa = st.sidebar.checkbox('Mostrar mapa', value=False)
	links = st.sidebar.checkbox('Mostar links', value=False)
	return no_display, mapa, links

def show_map(dataframe):
    ap_coord = []
    for key, values in dataframe.iterrows():
        lat, lon = values['latitude'], values['longitude']
        ap_coord.append([float(lat), float(lon)])
    ap_coord = pd.DataFrame(ap_coord, columns=['latitude', 'longitude'])
    st.map(ap_coord, zoom=11)

def format_ranked_list(df):
	lista = []
	for key, values in df.iterrows():
		lista.append(str(f'{key:02}')+'-----<a href={}>{}</a>-----{}'.format(values['apartment_link'], values['id'], round(values['score'], 3)))
	return lista


# <a href=[link]>[id]</a>

def main():
	select_block_container_style()
	no_display, mapa, links = set_sidebar()
	st.title('Recomendador de Apartamentos')
	st.header('Apartamentos')
	lgbm, rf, enc = set_variables()
	df = pd.read_csv('./dataset.csv')
	db_date = datetime.strptime(df['update_time'].unique()[0], '%Y-%m-%d')
	today = datetime.strptime(str(date.today()), '%Y-%m-%d')
	if ((today-db_date).days>7):
		with st.spinner('Atualizando os dados, e como é necessário pegar os dados do QuintoAndar, pode demorar uns minutos...'):
			update_db()
			df = pd.read_csv('./dataset.csv')
	dataset = encode_variables(enc, df)
	data = format_data(df, dataset, rf, lgbm)
	st.write(data[['id', 'area', 'custo', 'mobiliado', 'bairro', 'estacao', 'distancia', 'score']].head(no_display))
	# st.markdown('<a href={}>{}</a>'.format(data['apartment_link'].head(no_display).values[0], data['id'].head(no_display).values[0]), unsafe_allow_html=True)
	if mapa:
		st.header('Mapa')
		show_map(data.head(no_display))
	if links:
		st.sidebar.header('Links')
		st.sidebar.markdown('<br>'.join(format_ranked_list(data.head(no_display))), unsafe_allow_html=True)

if __name__ == '__main__':
	main()