import pandas as pd
import numpy as np
import requests as rq
from time import sleep
import os
from geopy import distance
from warnings import filterwarnings
from datetime import date

pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
filterwarnings("ignore")

def get_coordinates():
    df = pd.read_csv('district.csv')
    latlon = []
    for key, value in df.iterrows():
        diff = 0.008
        lat = value['Latitude']
        lon = value['Longitude']
        latlon.append([str(round(lat-diff,2))+','+str(round(lon+diff,2)),str(round(lat+diff,2))+','+str(round(lon-diff,2))])    
    df = df.merge(pd.DataFrame(latlon), how='left', left_index=True, right_index=True)
    df.columns = ['District', 'Latitude', 'Longitude', 'Coord_0', 'Coord_1']	
    return df

def get_district(coord_0, coord_1):
    url = 'https://www.quintoandar.com.br/api/yellow-pages/search?q=(and%20tipo:%27Apartamento%27(and%20(and%20for_rent:%27true%27)))&fq=local:[%27{}%27,%27{}%27]&return=id,local,aluguel,area,quartos,custo,endereco,regiao_nome,special_conditions,listing_tags,tipo,promotions,condo_iptu,vagas,amenidades&size=1000&q.parser=structured'
    furl = url.format(coord_1, coord_0)
    response = rq.get(furl).json()
    apes = []
    for hit in response['hits']['hit']:
        ap_id = hit.get('id')
        quartos = hit.get('fields').get('quartos')
        area = hit.get('fields').get('area')
        custo = hit.get('fields').get('custo')
        vagas = hit.get('fields').get('vagas')
        amenidades = hit.get('fields').get('amenidades')
        if 'NaoMobiliado' in amenidades:
            mobiliado = 0
        else:
            mobiliado = 1
        local = hit.get('fields').get('local')
        bairro = hit.get('fields').get('regiao_nome')
        endereco = hit.get('fields').get('endereco')
        apes.append([ap_id, quartos, area, custo, vagas, mobiliado, local, bairro, endereco])
    apes = pd.DataFrame(apes, columns=['id', 'quartos', 'area', 'custo', 'vagas', 'mobiliado',
                                       'local', 'bairro', 'endereco'])
    return apes

def get_apartments(district):
    apartments = pd.DataFrame()
    for key, value in district.iterrows():
        apartments = pd.concat([apartments, get_district(value['Coord_0'], value['Coord_1'])], ignore_index=True)
    return apartments

def get_datasets():
    district = get_coordinates()
    apartments = get_apartments(district)
    apartments.drop_duplicates(inplace=True)
    apartments.to_csv('tmp.csv', index=False)
    apartments = pd.read_csv('tmp.csv')
    df = pd.read_csv('dataset.csv')
    dados = df.merge(apartments[['id', 'local']], how='outer', left_on='id', right_on='id')
    metro = pd.read_csv('metrosp_stations_v2.csv').drop('Unnamed: 0', axis=1)
    return df, apartments, dados, metro

def clean_old_apes(df, dados):
    idx = dados[dados['local'].isnull()]['id'].values
    df = df.query('id not in @idx')
    return df

def new_apes(apartments, dados):
    idx = dados[dados['custo'].isnull()]['id'].values
    apartments = apartments.query('id in @idx')
    return apartments

def distancia(lat, lon):
    metro = pd.read_csv('metrosp_stations_v2.csv').drop('Unnamed: 0', axis=1)
    dist = []
    for idx, value in metro.iterrows():
        dist.append(distance.distance((metro.iloc[idx]['lat'],metro.iloc[idx]['lon']), (lat,lon)).km)
    min_dist = min(dist)
    estacao = metro.iloc[dist.index(min_dist)]['name']
    return min_dist, estacao

def get_subway_line(linha, apartments):
    lista = []
    for idx, value in apartments.iterrows():
        tmp = value['line'].split(', ')
        if linha in tmp:
            lista.append(1)
        else:
            lista.append(0)
    return lista

def process_new_apes(apartments, metro):
    apartments['latitude'] = apartments['local'].apply(lambda x: x.split(',')[0])
    apartments['longitude'] = apartments['local'].apply(lambda x: x.split(',')[1])
    apartments.drop('local', axis=1, inplace=True)
    apartments['metros'] = apartments.apply(lambda x: distancia(x['latitude'], x['longitude']), axis=1)
    apartments['estacao'] = apartments['metros'].apply(lambda x: x[1])
    apartments['distancia'] = apartments['metros'].apply(lambda x: x[0])
    apartments['distancia'] = apartments['distancia'].apply(lambda x: round(x, 2))
    apartments.drop('metros', axis=1, inplace=True)
    apartments = (apartments
                  .merge(metro[['name', 'line']], how='left', left_on='estacao', right_on='name')
                  .drop('name', axis=1))
    apartments['line'] = apartments['line'].apply(lambda x: x[1:-1])
    apartments['line'] = apartments['line'].apply(lambda x: x.replace("'", ''))
    linhas = ['amarela', 'azul', 'lilas', 'prata', 'verde', 'vermelha']
    for linha in linhas:
        apartments['linha_{}'.format(linha)] = get_subway_line(linha, apartments)
    apartments.drop('line', axis=1, inplace=True)
    apartments['update_time'] = date.today()
    site = 'https://www.quintoandar.com.br/imovel/'
    apartments['apartment_link'] = apartments.apply(lambda x: site+str(x['id']), axis=1)
    return apartments

def update_db():
    df, apartments, dados, metro = get_datasets()
    df = clean_old_apes(df, dados)
    apartments = new_apes(apartments, dados)
    apartments = process_new_apes(apartments, metro)
    dados = pd.concat([df, apartments], ignore_index=True)
    dados['update_time'] = date.today()
    dados.to_csv('dataset.csv', index=False)