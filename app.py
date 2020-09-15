import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import keras
from streamlit_folium import folium_static

import streamlit.components.v1 as components

import folium
from folium import plugins
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.write("""
# Eesti Statistika Rent Prediction App
This app predicts the apartment and house rents!

""")

st.sidebar.header('User Input Features')

#st.sidebar.markdown("""
#[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
#""")

# Collects user input features into dataframe

def user_input_features():
    latitude = st.sidebar.text_input("Latitude (Please input the value from the map)", 58.5953)
    longitude=st.sidebar.text_input("Longitude (Please input the value from the map)", 25.0136)
    county = st.sidebar.selectbox('county',('ida-virumaa', 'harjumaa', 'tartumaa', 'lääne-virumaa', 'pärnumaa',
       'võrumaa', 'viljandimaa', 'läänemaa', 'valgamaa', 'jõgevamaa',
       'raplamaa', 'saaremaa', 'järvamaa', 'põlvamaa'))
    
    rooms = st.sidebar.slider('Rooms', 1, 20, 1)
    area = st.sidebar.slider('Area, for apartment (sqm)', 5 ,1000, 44)

    total_area = st.sidebar.slider('Total Area, for house (sqm)', 0,10000,0)
    floor = st.sidebar.slider('Floor', 0,30,0)
    built_year = st.sidebar.slider('Built Year', 1422, 2020, 2020)
    bills_summer = st.sidebar.slider('Bills Summer', 6, 350, 50)
    bills_winter = st.sidebar.slider('Bills Winter', 6, 400, 100)

    sauna = st.sidebar.checkbox("Sauna")
    bathtub = st.sidebar.checkbox("Bathtub")
    garage = st.sidebar.checkbox("Garage")
    balcony = st.sidebar.checkbox("Balcony")
    lift = st.sidebar.checkbox("Lift")

    condition = st.sidebar.selectbox('Seisukord',('Condition_0', 'Condition_Heas korras','Condition_Keskmine','Condition_Renoveeritud', 'Condition_San. remont tehtud', 'Condition_Uus', 'Condition_Vajab renoveerimist', 'Condition_Vajab san. remonti', 'Condition_Valmis'))
    energy_mark = st.sidebar.selectbox('Energiamärgis',('0','A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'Puudub'))


    parking = st.sidebar.selectbox('Parking',('No Parking','Parking Maja', 'Parking Tasuline', 'Parking Tasuta', 'Parking Aia', 'Parking Ajavahemikul'))
    
    kitchen = st.sidebar.checkbox("Tahan täielikult varustatud kööki")
    heating = st.sidebar.checkbox("Heating")

    data = {
            'latitude': latitude,
            'longitude': longitude,
            'county': county,
            'rooms': rooms,
            'area': area,
            'total_area': total_area,
            'floor': floor,
            'builtyear': built_year,
            'bills_summer': bills_summer,
            'bills_winter': bills_winter,
            'sauna': sauna,
            'bathtub': bathtub,
            'garage':garage,
            'balcony':balcony,
            'lift':lift,
            'condition':condition,
            'energy_mark':energy_mark,
            'parking':parking,
            'kitchen':kitchen,
            'heating':heating

            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
#penguins_raw = pd.read_csv('penguins_cleaned.csv')
#penguins = penguins_raw.drop(columns=['species'])

#df=pd.DataFrame(0, index=1, columns=40)
df = input_df
#inp_usr=df.values
#x_test=[0] * 40

#x_test[1:15]=df.iloc[:, 1:16].to_numpy()


#if(x_test[0]=='Harju'):

df[['county_harjumaa', 'county_ida-virumaa',
       'county_järvamaa', 'county_jõgevamaa', 'county_lääne-virumaa',
       'county_läänemaa', 'county_pärnumaa', 'county_põlvamaa',
       'county_raplamaa', 'county_saaremaa', 'county_tartumaa',
       'county_valgamaa', 'county_viljandimaa', 'county_võrumaa',]] = pd.DataFrame([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]], index=df.index)


if(df.iloc[0]['county']=='lääne-virumaa'):
    df['county_lääne-virumaa']=1

if(df.iloc[0]['county']=='harjumaa'):
    df['county_harjumaa']=1

if(df.iloc[0]['county']=='ida-virumaa'):
    df['county_ida-virumaa']=1

if(df.iloc[0]['county']=='tartumaa'):
    df['county_tartumaa']=1


if(df.iloc[0]['county']=='raplamaa'):
    df['county_raplamaa']=1

if(df.iloc[0]['county']=='valgamaa'):
    df['county_valgamaa']=1

if(df.iloc[0]['county']=='võrumaa'):
    df['county_võrumaa']=1

if(df.iloc[0]['county']=='läänemaa'):
    df['county_läänemaa']=1


if(df.iloc[0]['county']=='pärnumaa'):
    df['county_pärnumaa']=1

if(df.iloc[0]['county']=='viljandimaa'):
    df['county_viljandimaa']=1

if(df.iloc[0]['county']=='jõgevamaa'):
    df['county_jõgevamaa']=1

if(df.iloc[0]['county']=='järvamaa'):
    df['county_järvamaa']=1



if(df.iloc[0]['county']=='saaremaa'):
    df['county_saaremaa']=1

if(df.iloc[0]['county']=='põlvamaa'):
    df['county_põlvamaa']=1



df=df.rename(columns={"location": "pricesqm"})
df['pricesqm']=9.41 #median pricesqm

df[['condition_0', 'condition_Heas korras', 'condition_Keskmine', 'condition_Renoveeritud', 'condition_San. remont tehtud', 'condition_Uus', 'condition_Vajab renoveerimist', 'condition_Vajab san. remonti', 'condition_Valmis']] = pd.DataFrame([[0,0,0,0,0,0,0,0,0]], index=df.index)
df[['en_mark_0', 'en_mark_A', 'en_mark_B', 'en_mark_C', 'en_mark_D', 'en_mark_E', 'en_mark_F', 'en_mark_G', 'en_mark_H', 'en_mark_puudub']] = pd.DataFrame([[0,0,0,0,0,0,0,0,0,0]], index=df.index)
df[['parking_0', 'parking_maja', 'parking_tasuline', 'parking_tasuta', 'parking_aia', 'parking_ajavahemikul']] = pd.DataFrame([[0,0,0,0,0,0]], index=df.index)



if (df.iloc[0]['condition']=='Condition_0'):
    df['condition_0']=1
if (df.iloc[0]['condition']=='Condition_Heas korras'):
    df['condition_Heas korras']=1
if (df.iloc[0]['condition']=='Condition_Keskmine'):
    df['condition_Keskmine']=1
if (df.iloc[0]['condition']=='Condition_Renoveeritud'):
    df['condition_Renoveeritud']=1
if (df.iloc[0]['condition']=='Condition_San. remont tehtud'):
    df['condition_San. remont tehtud']=1

if (df.iloc[0]['condition']=='Condition_Uus'):
    df['condition_Uus']=1

if (df.iloc[0]['condition']=='Condition_Vajab renoveerimist'):
    df['condition_Vajab renoveerimist']=1
if (df.iloc[0]['condition']=='Condition_Vajab san. remonti'):
    df['condition_Vajab san. remonti']=1
if (df.iloc[0]['condition']=='Condition_Valmis'):
    df['condition_Valmis']=1





if (df.iloc[0]['energy_mark']=='0'):
    df['en_mark_0']=1
if (df.iloc[0]['energy_mark']=='A'):
    df['energy_mark_A']=1
if (df.iloc[0]['energy_mark']=='B'):
    df['en_mark_B']=1
if (df.iloc[0]['energy_mark']=='C'):
    df['en_mark_C']=1
if (df.iloc[0]['energy_mark']=='D'):
    df['en_mark_D']=1

if (df.iloc[0]['energy_mark']=='E'):
    df['en_mark_E']=1

if (df.iloc[0]['energy_mark']=='F'):
    df['en_mark_F']=1
if (df.iloc[0]['energy_mark']=='G'):
    df['en_mark_G']=1
if (df.iloc[0]['energy_mark']=='H'):
    df['en_mark_H']=1
if (df.iloc[0]['energy_mark']=='Puudub'):
    df['en_mark_puudub']=1


if (df.iloc[0]['parking']=='No Parking'):
    df['parking_0']=1
if (df.iloc[0]['parking']=='Parking Maja'):
    df['parking_maja']=1
if (df.iloc[0]['parking']=='Parking Tasuline'):
    df['parking_tasuline']=1
if (df.iloc[0]['parking']=='Parking Tasuta'):
    df['parking_tasuta']=1
if (df.iloc[0]['parking']=='Parking Aia'):
    df['parking_aia']=1
if (df.iloc[0]['parking']=='Parking Ajavahemikul'):
    df['parking_ajavahemikul']=1

df=df.drop(['parking', 'energy_mark', 'condition'], axis=1)

df=df.rename(columns={"kitchen": "kitchen_1", "heating": "heating_1"})

column_titles = ['pricesqm', 'lat', 'long', 'rooms', 'area', 'total_area', 'floor',
       'builtyear', 'bills_summer', 'bills_winter', 'sauna', 'bathtub',
       'garage', 'balcony', 'lift', 'county_harjumaa', 'county_ida-virumaa',
       'county_järvamaa', 'county_jõgevamaa', 'county_lääne-virumaa',
       'county_läänemaa', 'county_pärnumaa', 'county_põlvamaa',
       'county_raplamaa', 'county_saaremaa', 'county_tartumaa',
       'county_valgamaa', 'county_viljandimaa', 'county_võrumaa',
       'condition_0', 'condition_Heas korras', 'condition_Keskmine',
       'condition_Renoveeritud', 'condition_San. remont tehtud',
       'condition_Uus', 'condition_Vajab renoveerimist',
       'condition_Vajab san. remonti', 'condition_Valmis', 
       'en_mark_0', 'en_mark_A', 'en_mark_B', 'en_mark_C', 'en_mark_D', 'en_mark_E', 'en_mark_F', 'en_mark_G', 'en_mark_H', 'en_mark_puudub', 'parking_0', 'parking_aia',
       'parking_ajavahemikul', 'parking_maja', 'parking_tasuline',
       'parking_tasuta', 'kitchen_1', 'heating_1']

df=df.reindex(columns=column_titles)
df.lat=input_df.latitude
df.long=input_df.longitude




# Displays the user input features
st.subheader('User Input features')





m = folium.Map(location=[58.5953, 25.0136], zoom_start=7)
popup=folium.LatLngPopup()
m.add_child(popup)
#x=folium.LatLngPopup().add_to(m)
#folium.ClickForMarker().add_to(m)
#popup_js_name = x.get_value()








folium_static(m)

#st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
st.write(df)


#latitude = st.sidebar.text_input("Latitude (Please input the value from the map)", 0)
    #longitude=st.sidebar.text_input("Longitude (Please input the value from the map)", 25.0136)


#st.write(ctypes.cast(id(m._children), ctypes.py_object).value)




# Reads in saved classification model
with open(f'randomforest.pkl', 'rb') as f:
    model = pickle.load(f)

# Apply model to make predictions

prediction = model.predict(df)



st.subheader('Prediction')

st.write(prediction)

st.subheader('Heatmap for Price per square meter in Estonia')

df_rent=pd.read_csv("df_rent.csv", sep=';' )
stationArr = df_rent[['lat', 'long', 'pricesqm']].values

# plot heatmap
m.add_children(plugins.HeatMap(stationArr, radius=15))
#components.iframe("""test.html""")


#folium_static(m)


st.markdown("<img src='https://g1.nh.ee/images/pix/file88261807_sa-logo.png' align='left' width=50%>", unsafe_allow_html=True)
