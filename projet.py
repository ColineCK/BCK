import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import pinv, svd
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from wordcloud import WordCloud
import seaborn as sns
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from streamlit_multipage import MultiPage
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import folium
from streamlit_folium import folium_static

from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from lazypredict.Supervised import LazyClassifier
import plotly.graph_objects as go
from streamlit_option_menu import option_menu


#Données
nom_fichier = "school-shootings-data.csv"

df =pd.read_csv(nom_fichier)
pd.set_option('display.max_rows', None) 

USA_map = "US_States3.json"

#=================================================================================
#                           Retraitement des données 
#=================================================================================

#Retraitement des données
df["date"] = pd.to_datetime(df['date'], format='%m/%d/%Y')

df["day_of_week"]=df["day_of_week"].astype(str)
#shooting_type
regroupement= {'indiscriminate': 'indiscriminate', 
               'targeted': 'targeted', 
               'accidental or targeted':'unclear', 
               'accidental':'accidental', 
               'targeted and indiscriminate':'targeted', 
               'hostage suicide':'suicide', 
               'public suicide':'suicide', 
               'public suicide (attempted)':'suicide', 
               'unclear':'unclear' }

df['shooting_type']  = df['shooting_type'].replace(regroupement)


df['time'] = pd.to_datetime(df["time"], format='%I:%M %p')

#time
# Définissez les plages horaires en format 24 heures
bins = pd.to_datetime(['00:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '23:59'], format='%H:%M')

# Créez une nouvelle colonne "Plage" en utilisant pd.cut
labels = ['00h00 - 8h00', '8h00 - 10h00', '10h00 - 12h00', '12h00 - 14h00', '14h00 - 16h00', '16h00 - 18h00', '18h00 - 00h00']
df['time'] = pd.cut(df['time'], bins=bins, labels=labels, right=False)


#================================================================================
#                               Introduction
#================================================================================

 
def introduction_page():
    df_work = df[["state", "casualties", "injured", "killed", "lat", "long", "school_name","date"]]
    df_work['lat'] = df_work['lat'].astype('int')
    df_work['long'] = df_work['long'].astype('int')
    df_workG = df[["state", "casualties"]]
    df_workG = df_work.groupby("state", as_index=False).sum()


    map = folium.Map(location=[48, -102], zoom_start=3)

    folium.Choropleth(
        geo_data=USA_map,
        name="choropleth",
        data=df_workG,
        columns=["state", "casualties"],
        key_on="feature.properties.NAME",
        nan_fill_color='white',
        nan_fill_opacity=0.4,
        fill_color="Reds",
        fill_opacity=0.7,
        line_opacity=.1,
        legend_name="Victimes",
    ).add_to(map)

    folium.LayerControl().add_to(map)

    # Ajoutez les marqueurs avec des info-bulles
    for index, row in df_work.iterrows():
        lat = float(row['lat'])
        long = float(row['long'])
        casualties = row['casualties']
        injured = row['injured']
        killed = row['killed']
        school_name = row['school_name']
        date = row['date']

        # Ajoutez un marqueur circulaire avec une info-bulle personnalisée
        folium.CircleMarker(
            location=[lat, long],
            radius=3,  # Ajustez la taille des marqueurs ici
            color='red',  # Changez la couleur en rouge
            fill=True,
            fill_color='red',
            fill_opacity=1,
            tooltip=f"Ecole: {school_name}<br>Victimes: {casualties}<br>Blessés: {injured}<br>Décés: {killed} <br>Date: {date}"  # Info-bulle avec toutes les informations
        ).add_to(map)

    # Utilisez la méthode folium_static de streamlit_folium pour afficher la carte
    folium_static(map)






#================================================================================
#                        Présentation des données
#================================================================================



def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    modify = st.checkbox("Ajouter des filtres")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrer la base de données sur : ", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Valeur pour {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Valeur pour {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valeur pour {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Valeur recherchée dans {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def pres_donnees():
    df_work = df
    df_work = df_work.drop(columns=['uid', 'nces_school_id', 'nces_district_id','state_fips', 'county_fips', 'ulocale' ])
    st.dataframe(filter_dataframe(df_work))
#==================================================================================
#                               Analyse univariée 
#==================================================================================

#------Fonctions :

def histo(variable, hauteur):
    df_work = df[[variable]].astype(float)
    df_work = df_work.fillna("")
    
    hist_chart = alt.Chart(df_work).mark_bar(color='darkred').encode(			
	x =alt.X(variable, bin=True, title=variable),
    y="count()"
	).properties(
		width=400,
		height=hauteur
	)
    return(hist_chart)
    
def barplot_vert(variable, hauteur):
	# Compter le nombre d'occurrences de chaque catégorie
	counts = df[variable].astype(str).value_counts().reset_index()
	counts.columns = [variable, 'Count']
	# Créez un graphique barplot pour la variable sélectionnée
	bar_chart = alt.Chart(counts).mark_bar(color='darkred').encode(			
		x=variable,
		y='Count'
	).properties(
		width=400,
		height=hauteur
	)
	return(bar_chart)
	
def barplot_hori(variable, hauteur):
	# Compter le nombre d'occurrences de chaque catégorie
	counts = df[variable].value_counts().reset_index()
	counts.columns = [variable, 'Count']
	# Créez un graphique barplot pour la variable sélectionnée
	bar_chart = alt.Chart(counts).mark_bar(color='darkred').encode(
		y=alt.Y(variable, title=variable, axis=alt.Axis(labelOverlap=True)),
    	x=alt.X("Count", title='Count')			
	).properties(
		width=400,
		height=hauteur
	)
	return(bar_chart)
    
def violin(variable):
    df_work = df[[variable]].astype(float)
    fig, ax = plt.subplots()
    sns.violinplot(data=df_work, y=variable, ax=ax, color = "darkred")

    return(fig)

	
def circulaire(variable):
	# Compter le nombre d'occurrences de chaque catégorie
    counts = df[variable].value_counts().reset_index()
    counts.columns = [variable, 'Proportion']
    # Créer un graphique circulaire avec Plotly Express
    fig = px.pie(counts, names=variable, color_discrete_sequence=px.colors.sequential.RdBu,  values='Proportion')
    fig.update_traces( textinfo = 'label+percent')
    fig.update_layout(autosize=False,width=400,height=400)
    fig.update_layout(showlegend=False)
    
    return(fig)

def graph_nuage_mots(variable):
    df_work = df[[variable]]
    df_work = df_work.fillna("")


    comment_words = ''
    mots_exclus = ["shooter", "a", "and", "who", "of", "found", "by", "in", "from", 
               "was", "open", "at", "gun", "shotgun", "issued", "puchased", "weapon","kept",
                  "mm", "ak", "ar", "a", "and", "san", "new", "e", "s","c"]
   
    # iterate 
    for val in df_work[variable]:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = mots_exclus,
                max_words=30,
                colormap="Reds",
		font_path="arial.ttf",
                min_font_size = 10).generate(comment_words)
    
 
    return(wordcloud)



def section_Univariee():
    # Ajoutez un widget pour choisir le type de graphique et la variable
    var_choix= st.selectbox('Sélectionnez un regroupement de variables :', ["Typologie de l'attaque", "Géographie","Ecole : identité et organisation", 'Temporalité',"Identité des attaquants"])
    c = st.container()
    
    # Style CSS pour l'encadré
    style_encadre = """
    	    padding: 10px;
	    border: 1px solid #800000;
	    border-radius: 5px;
	    background-color:  #ffcccc;
    """
    
    # Afficher le texte fixe avec encadré
    if var_choix == "Ecole : identité et organisation":
        # Texte fixe à afficher
        texte_fixe = """ Les écoles ciblées semblent de tous grades : primaire, collège, lycée. 
        Pour autant les écoles dites publiques sont 15 fois plus visées que les écoles privées.
        Enfin dans trois quarts des écoles ciblées il n'y avait pas de présence de force de protection au moment de l'attaque."""
        st.markdown(f'<div style="{style_encadre}">{texte_fixe}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        titre='**Noms des écoles concernées par les fusillades**'
        col1.markdown(titre)
        col1.image(graph_nuage_mots("school_name").to_array(), use_column_width=True)
        
        titre="**Type d'écoles impliquées dans une fusillade**"
        col2.markdown(titre)
        col2.altair_chart(barplot_vert("school_type", 400), use_container_width=True)
        
        col1, col2 = st.columns(2)
                
        titre = "**Nombre d'inscrits dans les écoles visées**"
        col1.markdown(titre)
        col1.altair_chart(histo("enrollment", 400), use_container_width=True) # changer en boxplot
        
        #lunch
        df["lunch_prop"] = (df['lunch']/df["enrollment"])
        titre= "**Proportion de boursiers au sein des établissements visés**"
        col2.markdown(titre)
        col2.altair_chart(histo("lunch_prop", 400), use_container_width=True) 
        
        col1, col2 = st.columns(2)
        
        #staffing
        df["staffing_prop"] = (df['staffing']/df["enrollment"])
        titre = "**Proportion de professeur par étudiant au sein des établissements visés**"
        col1.markdown(titre)
        col1.altair_chart(histo("staffing_prop", 400), use_container_width=True) 
        
        #ressource officer
        titre="**Présence/ abscence d'une force de protection dans les étabilssements visés**"
        df['ind_officer'] = np.where(df['resource_officer'] > 0, 'Présence', 'Absence')
        col2.markdown(titre)
        col2.plotly_chart(circulaire("ind_officer"), use_container_width=True)
        
    elif var_choix == "Typologie de l'attaque": # barplot ou histogramme ? ajouter boxplot
        texte_fixe = """Environ trois quarts des fusillades recensées ne sont pas meurtières. Ainsi plus de 90% des attaques engendrent moins de 5 victimes (blessés ou tués). 
        Une grande proportion des fusillades sont préméditées et les armes utilisées sont majoritairement des armes à feu.
                        Les armes utilisées sont pour la majorité déclarées, et appartiennent le plus souvent à l'agresseur ou à un proche de celui-ci.
                        """
        st.markdown(f'<div style="{style_encadre}">{texte_fixe}</div>', unsafe_allow_html=True)
    
        col1, col2 = st.columns(2)
        titre="**Proportions d'attaques meurtrières**"
        col1.markdown(titre)
        df['ind_killed'] = np.where(df['killed'] > 0, 'Meutrières', 'Non meurtrières')
        col1.plotly_chart(circulaire("ind_killed"), use_container_width=True)
        
        titre = "**Répartition du nombre de victimes totales**"
        col2.markdown(titre)
        col2.altair_chart(histo("casualties", 400), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        titre = "**Répartition du nombre de blessés**"
        col1.markdown(titre)
        col1.pyplot(violin("injured"), use_container_width=True)
        
        titre = "**Répartition du nombre de morts**"
        col2.markdown(titre)
        col2.pyplot(violin("killed"), use_container_width=True)
        
        titre="**Type de la fusillade**"
        st.markdown(titre)
        st.altair_chart(barplot_vert("shooting_type", 400), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        titre ="**Armes utilisées**"
        col1.markdown(titre)
        col1.image(graph_nuage_mots("weapon").to_array(), use_column_width=True)
        
        titre ="**Provenance des armes**"
        col2.markdown(titre)
        col2.image(graph_nuage_mots("weapon_source").to_array(), use_column_width=True)
        
    elif var_choix == "Temporalité":
        
        texte_fixe = """ Il semble y avoir une forte croissance du nombre d'attaques depuis 2016. 
        Aucun jour ni aucune heure ne semble privilégié par les assaillants."""
        st.markdown(f'<div style="{style_encadre}">{texte_fixe}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        titre = "**Nombre de fusillades par année**"
        col1.markdown(titre)
        col1.altair_chart(barplot_vert("year", 400),use_container_width=True)
        
        titre = "**Nombre de fusillades par année scolaire**"
        col2.markdown(titre)
        col2.altair_chart(barplot_hori("school_year",1300), use_container_width=True)
        
        titre = "**Répartition des fusillades dans la semaine**"
        col1.markdown(titre)
        col1.plotly_chart(circulaire("day_of_week"), use_container_width=True)
        
        titre = "**Réparitition de l'horaire des fusillades**"
        col1.markdown(titre)
        col1.plotly_chart(circulaire("time"), use_container_width=True)
        
    elif var_choix == "Géographie":
        
        texte_fixe = """Il est observé une grande disparité du nombre de fusillades entre les États.
        L'État de Californie est le plus touché avec 40 attaques en milieu scolaire, vient ensuite le Texas et la Caroline du Nord. 
        En revanche le Wyoming ne semble pas avoir été touché en 25 ans.
        Ainsi la ville de Los Angeles s'impose comme la ville la plus meurtrie par de telles attaques."""
        st.markdown(f'<div style="{style_encadre}">{texte_fixe}</div>', unsafe_allow_html=True)
        
        titre = '**Nombre de fusillades par État**'
        st.markdown(titre)
        st.altair_chart(barplot_hori("state", 1200), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        titre ='**Comtés concernés par les fusillades**'
        col1.markdown(titre)
        col1.image(graph_nuage_mots("county").to_array(), use_column_width=True)
        
        titre = '**Villes concernées par les fusillades**'
        col2.markdown(titre)
        col2.image(graph_nuage_mots("city").to_array(), use_column_width=True)
        
    elif var_choix == "Identité des attaquants":
        texte_fixe = """ Le profil type d'un assaillant d'un établissement américain est un homme, de 25 ans, étudiant dans l'établissement.
        Dans la majorité des cas il est capturé vivant."""
        st.markdown(f'<div style="{style_encadre}">{texte_fixe}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        #age 
        titre = '**Répartition âge des attaquants**'
        col1.markdown(titre)
        col1.altair_chart(histo("age_shooter1", 400), use_container_width=True)
            
        #Sexe
        titre = "**Sexe de l'attaquant**"
        df['ind_sexe_att'] = np.where(df['gender_shooter1'] == "m", 'Homme', 'Femme')
        col2.markdown(titre)
        col2.plotly_chart(circulaire("ind_sexe_att"), use_container_width=True)
        
        c2 = st.container()
        #Mortalité
        titre = "**Etat de l'attaquant (après attaque)**"
        df['ind_mort_attaquant'] = np.where(df['shooter_deceased1'] == 1, 'Décès', 'Vivant')
        col1.markdown(titre)
        col1.plotly_chart(circulaire("ind_mort_attaquant"), use_container_width=True)
        
        #relationship
        titre = "**Relation de l'attaquant avec l'établissement**"
        col2.markdown(titre)
        col2.image(graph_nuage_mots("shooter_relationship1").to_array(), use_column_width=True)


#==================================================================================
#                               Analyse multivariée
#==================================================================================
#pip install branca==0.3.1  si ne fonctionne pas
   
# Ajouter graphique en dessous avec l'evolution du nombre de victimes totales, blessés et morts par       
# tester cirlce marker
def biv_map( date_debut, date_fin):
    df_work = df[df['year'] >= date_debut ]
    df_work = df_work[df_work['year'] <= date_fin ]
    df_work = df_work[["state","casualties", "injured", "killed"]]
    df_work = df_work.groupby("state", as_index=False).sum()

    map = folium.Map(location=[48, -102], zoom_start=3)
            
    folium.Choropleth(
        geo_data=USA_map,
        name="choropleth",
        data=df_work,
        columns=["state", "casualties"],
        key_on="feature.properties.NAME",
        nan_fill_color='white',
        nan_fill_opacity=0.4,
        fill_color="Reds",
        fill_opacity=0.7,
        line_opacity=.1,
        legend_name="Victimes",
    ).add_to(map)
        
    folium.LayerControl().add_to(map)
    folium_static(map)
    
def graph(date_debut, date_fin):
    df_work = df[df['year'] >= date_debut ]
    df_work = df_work[df_work['year'] <= date_fin ]
    df_work = df_work[["year","casualties", "injured", "killed"]]
    df_work = df_work.groupby("year", as_index=False).sum()
    st.line_chart(df_work,  x = "year", y = ["casualties", "injured","killed"])
    



def graph_state(date_debut, date_fin, variable, states):
    df_work = df[df['year'] >= date_debut ]
    df_work = df_work[df_work['year'] <= date_fin ]
    df_work = df_work[["year", "state","casualties", "injured", "killed"]]
    df_work = df_work.groupby(["year", "state"],as_index=False)["casualties", "injured", "killed"].sum()
    
    st.write("Vous avez sélectionné: {}".format(", ".join(states)))
    dfs = {state: df_work[df_work["state"] == state] for state in states}
    fig = go.Figure()
    for state, d in dfs.items():
        fig = fig.add_trace(go.Scatter(x=d["year"], y=d[variable], name=state))
    st.plotly_chart(fig) 
    
        
    

def section_bivariee():
    st.write("Dans cette partie est réalisée une étude de dépendance entre le nombre de vitimes, blessés ou morts et la situation géographique de l'école visée ou la date.")
    values = st.slider('Sélectionnez les années que vous souhaitez étudier : ',1999, 2023,(1999,2023))
    
    st.subheader("**Évolution du nombre de victimes au cours du temps**")
    graph( values[0], values[1])
    
    st.subheader("**Cartographie des victimes sur les années sélectionnées**")
    biv_map( values[0], values[1])
    st.markdown("\n")
    st.subheader("**Comparaison des États**")
    clist = df["state"].unique().tolist()
    states = st.multiselect("Sélectionnez des États : ", clist)
    st.write("**Comparaison sur la variable nombre de victimes sur les années :**")
    graph_state(values[0], values[1], "casualties",states)
    st.write("**Comparaison sur la variable nombre de tués sur les années :**")
    graph_state(values[0], values[1], "killed", states)
    st.write("**Comparaison sur la variable nombre de blessés sur les années :**")
    graph_state(values[0], values[1], "injured", states)
    


   
#==========================================================================================
#					Machine learning
#=========================================================================================== 




dfML = pd.read_csv(nom_fichier)

selected_columns = ['age_shooter1', 'resource_officer', \
	            'gender_shooter1','shooter_relationship1', 'weapon', \
	            'school_type', 'enrollment', 'staffing','lunch', \
	             'year',
	             'state','shooting_type','shooter_deceased1','lat','long']		             

	
#========================= Retraitements=======================================
	

#Casualties
dfML['casualties'] = dfML['casualties'].apply(lambda x: 0 if x == 0 else 1)
#Killed
dfML['killed'] = dfML['killed'].apply(lambda x: 0 if x == 0 else 1)

#injured
dfML['injured'] = dfML['injured'].apply(lambda x: 0 if x == 0 else 1)

#gender_shooter1
dfML['gender_shooter1'] = dfML['gender_shooter1'].apply(lambda x: 0 if x == "m" else 1)

#school_type
dfML['school_type'] = dfML['school_type'].apply(lambda x: 0 if x == "public" else 1)

#shooter_relationship1
dfML['shooter_relationship1'] = dfML['shooter_relationship1'].apply(lambda x: 0 if x == "student" else 1)

#weapon	
dfML['weapon'] = dfML['weapon'].apply(lambda x: 0 if pd.notna(x) and isinstance(x, str) and (('pistol' in x or 'handgun' in x or 'revolver' in x) and not ('rifle' in x or 'automatic' in x)) else 1)

#state
regroupement_etat = {}
state_numerique = []
state = dfML['state']
# Parcourir la liste des États
for etat in state:
# Si l'État n'est pas déjà dans le dictionnaire, ajoutez-le avec un nouveau numéro
	if etat not in regroupement_etat:
		regroupement_etat[etat] = len(regroupement_etat) + 1
		# Ajouter le numéro correspondant à la liste state_numerique
		state_numerique.append(regroupement_etat[etat])

dfML['state'] =dfML['state'].replace(regroupement_etat)    
		                            
#variable shooting_type                                          
regroupement_shooting= {'indiscriminate': 2, 
	       'targeted': 2, 
	       'accidental or targeted':3, 
	       'accidental':4, 
	       'targeted and indiscriminate': 2,
	       'hostage suicide':5, 
	       'public suicide':5, 
	       'public suicide (attempted)':5, 
	       'unclear':3 }

dfML['shooting_type']  = dfML['shooting_type'].replace(regroupement_shooting)
	




def ML_Best_models(cible, dfML):
	#Trois choix pour cible : casualties, killed, injured
	dfBM = dfML.dropna(subset=selected_columns+[cible])
	X = dfBM[selected_columns]
	X = pd.get_dummies(X,drop_first=False)
	y = dfBM[cible]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	#proportion de 0 et de 1 dans cible
	value_counts = y_test.value_counts()
	proportion_0 = value_counts[0] / y_test.shape[0]
	proportion_1 = value_counts[1] / y_test.shape[0]
	
	#meilleurs modèles
	reg = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
	models, predictions = reg.fit(X_train, X_test, y_train, y_test)
	models
	
	if (cible =="casualties"):
		st.write("Dans le jeu 'test' ", "{:.2f}".format(proportion_1*100) ,"% des fusillades générent au moins une victime. Cela signifie que le modèle nul qui prédit '1' peu importe l'entrée, prédit correctement dans ", "{:.2f}".format(proportion_1*100) ,"% des cas. Un tel modèle est donc meilleur que la majorité des modèles présentés ci-dessus.")
	if (cible =="injured"):
		st.write("Dans le jeu 'test' ", "{:.2f}".format(proportion_1*100),"% des fusillades générent au moins un bléssé. Cela signifie que le modèle nul qui prédit '1' peu importe l'entrée, prédit correctement dans ", "{:.2f}".format(proportion_1*100) ,"% des cas. Un tel modèle est donc meilleur que la majorité des modèles présentés ci-dessus.")
	if (cible =="killed"):
		st.write("Dans le jeu 'test' ", "{:.2f}".format(proportion_1*100),"% des fusillades générent au moins un mort. Cela signifie que le modèle nul qui prédit '1' peu importe l'entrée, prédit correctement dans ", "{:.2f}".format(proportion_1*100) ,"% des cas. Un tel modèle est donc meilleur que la majorité des modèles présentés ci-dessus.")

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def ML_SVM_2D(dfML,y,x1,x2):

	selected_columns_2D = [x1,x2 ]  # variables pour le graphe 2D
	dfML = dfML.dropna(subset=selected_columns+[y])
	
	# Dataset for decision function visualization: we only keep the first two
	# features in X and sub-sample the dataset to keep only 2 classes and
	# make it a binary classification problem.
	X = dfML[selected_columns]
	X = pd.get_dummies(X,drop_first=False)
	y = dfML[y]
	X_2d = X[selected_columns_2D]
	y_2d = y


	# It is usually a good idea to scale the data for SVM training.
	# We are cheating a bit in this example in scaling all of the data,
	# instead of fitting the transformation on the training set and
	# just applying it on the test set.

	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	X_2d = scaler.fit_transform(X_2d)


	# Train classifiers
	# -----------------
	#
	# For an initial search, a logarithmic grid with basis
	# 10 is often helpful. Using a basis of 2, a finer
	# tuning can be achieved but at a much higher cost.

	C_range = np.logspace(-2, 10, 13)
	gamma_range = np.logspace(-9, 3, 13)
	param_grid = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
	grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
	
	
	grid.fit(X, y)
	print("Les variables explicatives sont : ","x1"," et ","x2.")
	print(
	    "Les meilleurs paramètres sont %s avec un score de %0.2f"
	    % (grid.best_params_, grid.best_score_)
	)


	# Now we need to fit a classifier for all parameters in the 2d version
	# (we use a smaller set of parameters here because it takes a while to train)

	C_2d_range = [1e-2, 1, 1e2]
	gamma_2d_range = [1e-1, 1, 1e1]
	classifiers = []
	for C in C_2d_range:
		for gamma in gamma_2d_range:
			clf = SVC(C=C, gamma=gamma)
			clf.fit(X_2d, y_2d)
			classifiers.append((C, gamma, clf))


	# Visualization
	# -------------
	#
	# draw visualization of parameter effects

	# Visualisation des deux graphiques côte à côte
	fig, axes = plt.subplots(1, 2, figsize=(16, 6))
	
	xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
	# Premier graphique (subplot 1)
	for k, (C, gamma, clf) in enumerate(classifiers):
		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		
		axes[0].set_title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)), size="medium")
		axes[0].pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
		axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors="none")
		axes[0].set_xticks(())
		axes[0].set_yticks(())
		axes[0].set_xlabel(x1)
		axes[0].set_ylabel(x2)
		axes[0].axis("tight")
		
	scores = grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))

	# Draw heatmap of the validation accuracy as a function of gamma and C

	plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
	im = axes[1].imshow(
		scores,
		interpolation="nearest",
		cmap=plt.cm.hot,
		norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
		)
	axes[1].set_xlabel("gamma")
	axes[1].set_ylabel("C")
	plt.colorbar(im, ax=axes[1])
	axes[1].set_xticks(np.arange(len(gamma_range)))
	axes[1].set_yticks(np.arange(len(C_range)))
	axes[1].set_xticklabels(gamma_range, rotation=45)
	axes[1].set_yticklabels(C_range)
	axes[1].set_title("Validation accuracy : ")
	
	st.pyplot(fig)

	
	



    
def ML_section():
	#Choix de la variable cible
	st.subheader("Étude de la performance de différents modèles")
	st.write("L'étude de performance est dans un premier temps réalisée en prenant en compte toutes les variables explicatives d'intérêt.")
	cible_choix= st.selectbox('Sélectionnez une variable cible :', ["Nombre de victimes", "Nombre de blessés", "Nombre de morts"])
	cible =""
	
	if cible_choix == "Nombre de victimes":
	    ML_Best_models("casualties", dfML)
	    cible = "casualties"
	elif cible_choix == "Nombre de blessés":
	    ML_Best_models("injured", dfML)
	    cible = "injured"
	elif cible_choix == "Nombre de morts":
	    ML_Best_models("killed", dfML)
	    cible = "killed"
			
	st.subheader("Focus sur le SVM")
	st.write("Par la suite, il a été choisi de se concentrer sur un modèle Support Vector Machine (SVM), réduit à deux variables explicatives pour des raisons de visualisation.")
	#Choix variables explicatives

	# Sélection de deux éléments
	selected_options = st.multiselect('Sélectionnez deux options:', selected_columns, default=selected_columns[:2])

	# Vérification pour s'assurer que l'utilisateur sélectionne exactement deux éléments
	if len(selected_options) > 2:
	    	st.warning('Veuillez sélectionner exactement deux options.')
	    	selected_options = st.multiselect('Sélectionnez deux options:', selected_columns, default=None)

	# Affichage des éléments sélectionnés
	st.write('Vous avez sélectionné:', selected_options[0] , ' et ', selected_options[1], '.')
	st.write("Les graphiques peuvent prendre quelques secondes à être affichés.")
	
	ML_SVM_2D(dfML,cible,selected_options[0],selected_options[1])
	
	st.write("Le graphique de droite fait référence au tunning des hyperparamètres : 'Gamma' le paramètre d'échelle de longueur du noyau et 'C' le paramètre de pénalisation. Ainsi cette étape intermédiaire permet en théorie d'avoir un meilleur ajustement du modèle. C'est pourquoi, le modèle SVM ici ne semble pas adéquat au vu de la faible variabilité de la précision du modèle en fonction des hyperparamètres.")
	st.write("Le graphique de gauche représente la classification de la variable cible en fonction des deux variables explicatives choisies. Ainsi la couleur bleu est attribuée à une attaque dont la variable cible est nulle, alors que la couleur rouge représente une attaque avec une variable cible strictement positive (au moins une victime, un blessé ou un mort)")
	
	st.title("Conclusion")
	st.write("Cette étude ménée sur des données concernant les fusillades aux États-Unis dans le milieu scolaire a permis de dégager un profil type de l'attaque : plutôt perpétrée par un homme, souvent ancien étudiant de l'école visée dans un État peuplé. Pour autant, les modèles de Machine Learning ne semble pas prévoir de manière satisfaisante le fait ou non d'avoir des victimes en fonction des caractéristiques d'une attaque. ")
	
	


	


#==========================================================================================
#                                   Application
#=========================================================================================
 

with st.sidebar:
    selected = option_menu("Sommaire", ["Introduction", 'Présentation des données', 'Analyse univariée', 'Analyse multivariée', 'Machine learning'], default_index=1)
    

if selected == "Introduction":
     st.title("Contexte")
     st.write("La fusillade de Columbine aux États-Unis le 20 avril 1999 a fait 13 morts et 24 blessés. Depuis cette date, chaque année des établissements scolaires américains sont touchés par des attaques. \n Est-il possible de dresser le portrait type d'une attaque en milieu scolaire aux Etats-Unis ?")
     introduction_page()
elif selected == 'Présentation des données':
    st.title("Présentation de la base de données")
    st.write("La base de données utilisée pour cette étude est éditée par le journal 'The Washington Post' et recense toutes les fusillades de masse aux États-Unis dans les écoles du premier et second cycle depuis la fusillade de Colombine. Elle comporte 387 observations. Vous pouvez accéder à la base de données via le lien suivant : https://github.com/washingtonpost/data-school-shootings .")
    
    st.write("Explorez la base de données en ajoutant des filtres : ")

    pres_donnees()
elif selected == 'Analyse univariée':
    st.title("Analyse univariée")
    st.write("Cette étude commence par réaliser un état des lieux de chaque variable. Celles-ci ont été regroupées en cinq catégories : Typologie de l'attaque, Géographie, Ecole : identité et organisation, Temporalité et Identité des attaquants.")
    section_Univariee()
elif selected == 'Analyse multivariée':
    st.title("Analyse multivariée")
    
    section_bivariee()
else:
    st.title("Machine Learning")
    st.write("Cette partie vise à essayer de contruire un modèle prédictif pour déterminer si une attaque engendre des victimes, des blessés ou des morts en fonction des caractéristiques de l'attaque.")
    ML_section()



