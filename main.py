import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopy
from geopy.geocoders import Nominatim


# Es guarden les sortides en un arxiu de text
sys.stdout = open('resultat.txt', 'w')

# Es carrega el conjunt de dades de l'arxiu CSV
data = pd.read_csv("Barcelona_rent_price.csv")

# Es mostren les primeres files del dataset per veure l'estructura
print(data.head())

unique_values = data['Average _rent'].unique()
print(unique_values[:10])  # Es mostren els 10 primers valors unics

value_counts = data['Average _rent'].value_counts()
print(value_counts)

sample_data = data.sample(10)  # Es seleccionen 10 filas al atzar
print(sample_data[['Average _rent', 'Price']])

data = data.replace([float('inf'), float('-inf')], pd.NA)

# Es creen noves columnes per al preu mensual i el preu per metre quadrat
data['Price_monthly'] = data.apply(lambda row: row['Price'] if row['Average _rent'] == 'average rent (euro/month)' else None, axis=1)
data['Price_per_m2'] = data.apply(lambda row: row['Price'] if row['Average _rent'] == 'average rent per surface (euro/m2)' else None, axis=1)

# Es crea una columna de data combinant l'any i el trimestre
def convert_to_date(row):
    year = row['Year']
    trimester = row['Trimester']
    if trimester == 1:
        month = 1  # Primer trimestre comença gener
    elif trimester == 2:
        month = 4  # Segon trimestre comença abril
    elif trimester == 3:
        month = 7  # Tercer trimestre comença juliol
    elif trimester == 4:
        month = 10 # Quart trimestre comença octubre
    return pd.Timestamp(year=year, month=month, day=1)

data['Date'] = data.apply(convert_to_date, axis=1)

print('Es mostren les primeres files del DataFrame per a verificar les noves columnes')
print(data.head())
print('Informació general de nou del dataset')
print(data.info())

# Configuració general de l'estil de Seaborn
sns.set(style="whitegrid")

# Evolució del preu mensual
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Price_monthly', data=data, estimator='mean', errorbar=None, color='dodgerblue', linewidth=2)
plt.title('Evolució del Preu de Lloguer Mensual a Barcelona (2014-2022)', fontsize=16)
plt.xlabel('Data', fontsize=14)
plt.ylabel('Preu (€)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("Evolució_Preu_Lloguer_Mensual_Barcelona.png")
plt.close()

# Evolució del preu per metre quadrat
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Price_per_m2', data=data, estimator='mean', errorbar=None, color='darkorange', linewidth=2)
plt.title('Evolució del Preu de Lloguer per Metre Quadrat a Barcelona (2014-2022)', fontsize=16)
plt.xlabel('Data', fontsize=14)
plt.ylabel('Preu (€/m2)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("Evolucio_Preu_Lloguer_per_metre_quadrat_Barcelona.png.png")
plt.close()

# Comparació de preus per districte usant preu mensual
plt.figure(figsize=(14, 8))
sns.lineplot(x='Date', y='Price_monthly', hue='District', data=data)
plt.title('Evolució del Preu de Lloguer Mensual per Districte a Barcelona')
plt.xlabel('Data')
plt.ylabel('Preu (€)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Distrito', fontsize=6, title_fontsize=8)
plt.savefig("Evolució del Preu de Lloguer Mensual per Districte a Barcelona.png")
plt.close()


# Comparació de preus per barri utilitzant el preu mensual 
top_neighbourhoods = data.groupby('Neighbourhood')['Price_monthly'].mean().nlargest(10).index
top_data = data[data['Neighbourhood'].isin(top_neighbourhoods)]

plt.figure(figsize=(14, 8))
sns.boxplot(x='Neighbourhood', y='Price_monthly', data=top_data)
plt.title("Comparació de Preus de Lloguer Mensual per Barri (Top 10)", fontsize=16)
plt.xlabel('Barri', fontsize=14)
plt.ylabel('Preu (€)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.savefig("Comparació de Preus de Lloguer Mensual per Barri (Top 10).png")
plt.close()

# Distribució del preu del lloguer mensual per trimestre
plt.figure(figsize=(14, 8))
sns.violinplot(x='Year', y='Price_monthly', hue='Trimester', data=data)
plt.title('Distribució del Preu de Lloguer Mensual per Trimestre a Barcelona (2014-2022)')
plt.xlabel('Any', fontsize=14)
plt.ylabel('Preu (€)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Distribució del Preu de Lloguer Mensual per Trimestre a Barcelona (2014-2022).png")
plt.close()

# Mapa de calor de preus mensuals mitjans de lloguer per barri i any
pivot_table = data.pivot_table(values='Price_monthly', index='Neighbourhood', columns='Year', aggfunc='mean')

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".1f")
plt.title('Mapa de calor de preus mensuals mitjans de lloguer per barri i any', fontsize=16)
plt.xlabel('Any', fontsize=14)
plt.ylabel('Barri', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Mapa de calor de preus mensuals mitjans de lloguer per barri i any.png")
plt.close()

# Es filtren les dades no nul·les de Price_monthly
data_monthly = data.dropna(subset=['Price_monthly'])

# Es crea el gràfic interactiu de línies
fig = px.line(data_monthly, x='Date', y='Price_monthly', title='Evolució del Preu de Lloguer Mensual a Barcelona (2014-2022)',
              labels={'Price_monthly': 'Preu (€)', 'Date': 'Data'})

# Es mostra el gràfic
fig.write_html("Evolucio_Preu_Lloguer_Mensual_Barcelona.html")

# S'obté una llista dels diferents districtes i barris de Barcelona
districts = data['District'].unique()
neighbourhoods = data['Neighbourhood'].unique()

# Es mostren els diferents districtes i barris
print("Distritos de Barcelona:")
print(districts)
print("\nBarrios de Barcelona:")
print(neighbourhoods)

# S'inicialitza l'objecte de Nominatim 
geolocator = Nominatim(user_agent="barcelona_explorer")

# Es crea un diccionari per a emmagatzemar les coordenades de longitud i latitud de cada barri
coordinates = {}

# S'itera sobre cada barri y s'obtenen les coordenades de longitud i latitud
for neighbourhood in neighbourhoods:
    location = geolocator.geocode(neighbourhood + ", Barcelona")
    if location:
        coordinates[neighbourhood] = (location.latitude, location.longitude)

# Es mostren les coordenades de longitud i latitud de cada barri
for neighbourhood, coords in coordinates.items():
    print(f"{neighbourhood}: {coords}")

# S'afegeixen les columnes de longitud i latitud al DataFrame data
data['Latitude'] = data['Neighbourhood'].map(lambda x: coordinates[x][0] if x in coordinates else None)
data['Longitude'] = data['Neighbourhood'].map(lambda x: coordinates[x][1] if x in coordinates else None)


print(data.head())

# Es filtren les dades per al trimestre 1 de l'any 2022 i on el preu mensual no sigui nul
data_trimestre1_2022 = data[(data['Year'] == 2022) & (data['Trimester'] == 1) & (data['Price_monthly'].notnull())]

# Es crea el gràfic del mapa interactiu
fig = px.scatter_mapbox(data_trimestre1_2022, 
                         lat="Latitude", 
                         lon="Longitude", 
                         hover_name="Neighbourhood", 
                         hover_data={"Latitude": False, "Longitude": False, "Price_monthly": True},
                         color="Price_monthly",  
                         color_continuous_scale=px.colors.sequential.Viridis,
                         size_max=15, 
                         zoom=10,
                         mapbox_style="carto-positron",
                         title="Preu Mig Mensual de Lloguer per Barri a Barcelona (Trimestre 1, 2022)")

# S'actualitza el diseny del mapa 
fig.update_layout(margin=dict(r=0, t=30, l=0, b=0))

# Es guarda el mapa interactiu com un arxiu HTML
fig.write_html("mapa_interactiu_trimestre1_2022.html")
                        
 # Es filtren les dades per al trimestre 2 de l'any 2022 i on el preu per metre quadrat no sigui nul
data_trimestre2_2022 = data[(data['Year'] == 2022) & (data['Trimester'] == 2) & (data['Price_per_m2'].notnull())]

# Es crea el gràfic del mapa interactiu
fig = px.scatter_mapbox(data_trimestre2_2022, 
                         lat="Latitude", 
                         lon="Longitude", 
                         hover_name="Neighbourhood", 
                         hover_data={"Latitude": False, "Longitude": False, "Price_per_m2": True},
                         color="Price_per_m2",  # Aquí se define el color en función del precio por metro cuadrado
                         color_continuous_scale=px.colors.sequential.Viridis,
                         size_max=15, 
                         zoom=10,
                         mapbox_style="carto-positron",
                         title="Preu Mitjà per Metre Quadrat de Lloguer per Barri a Barcelona (Trimestre 2, 2022)")

# S'actualitza el disseny del mapa
fig.update_layout(margin=dict(r=0, t=30, l=0, b=0))

# Es guarda el mapa interactiu com un arxiu HTML 
fig.write_html("mapa_interactiu_trimestre2_2022.html")     

# Restauració del output estàndar a consola 
sys.stdout.close()
sys.stdout = sys.__stdout__