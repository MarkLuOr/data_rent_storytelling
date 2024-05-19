import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# Es guarden les sortides en un arxiu de text
sys.stdout = open('resultat.txt', 'w')

# Es carrega el conjunt de dades de l'arxiu CSV
data = pd.read_csv("Barcelona_rent_price.csv")

# Es mostren les primeres files del dataset per veure l'estructura
print(data.head())

unique_values = data['Average _rent'].unique()
print(unique_values[:10])  # Mostrar los primeros 10 valores únicos

value_counts = data['Average _rent'].value_counts()
print(value_counts)

sample_data = data.sample(10)  # Seleccionar 10 filas al azar
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
        month = 1  # Primer trimestre comienza en enero
    elif trimester == 2:
        month = 4  # Segundo trimestre comienza en abril
    elif trimester == 3:
        month = 7  # Tercer trimestre comienza en julio
    elif trimester == 4:
        month = 10 # Cuarto trimestre comienza en octubre
    return pd.Timestamp(year=year, month=month, day=1)

data['Date'] = data.apply(convert_to_date, axis=1)

print('Es mostren les primeres files del DataFrame per a verificar les noves columnes')
print(data.head())
print('Informació general de nou del dataset')
print(data.info())

# Configuració ngeneral de l'estil de Seaborn
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



# Restauració del output estàndar a consola 
sys.stdout.close()
sys.stdout = sys.__stdout__