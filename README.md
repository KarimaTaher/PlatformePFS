
---

# **Chapitre 1 : Introduction Générale**

## **Contexte du Projet**

Le pétrole représente plus de 30 % de la consommation énergétique mondiale et demeure l’un des produits les plus échangés à l’échelle internationale. Aux États-Unis, premier producteur mondial depuis 2018, le pétrole joue un rôle crucial dans la croissance économique, la politique étrangère et la sécurité énergétique. Toutefois, les prix du pétrole sont notoirement instables, influencés par une diversité de facteurs tels que les événements géopolitiques, les décisions de l’OPEP, les fluctuations de la demande, les catastrophes naturelles ou encore les innovations technologiques.

Depuis la crise financière de 2008 jusqu’à la pandémie de COVID-19 en 2020, le baril a connu de grandes variations, passant parfois de plus de 100 USD à moins de 20 USD en quelques mois. Cette instabilité perturbe les politiques économiques, déstabilise les marchés financiers et affecte directement les emplois dans les secteurs liés à l’énergie.

Dans ce contexte, disposer d’un outil capable d’**anticiper les variations du prix du pétrole**, même approximativement, devient un enjeu stratégique majeur pour les gouvernements, les entreprises, les investisseurs et les citoyens.

---

## **Problématique**

Malgré de nombreuses recherches et tentatives de modélisation, prédire avec précision les prix du pétrole reste extrêmement complexe. Cette difficulté provient d’une combinaison de facteurs imprévisibles : déséquilibres entre l’offre et la demande, tensions géopolitiques, évolutions réglementaires, spéculations financières, etc.

Une mauvaise prévision peut entraîner des décisions économiques inadaptées, une mauvaise gestion énergétique ou encore d’importantes pertes financières. Dès lors, une question essentielle se pose :

> **Peut-on développer une solution d’intelligence artificielle capable de prédire de manière fiable les prix du pétrole, et si oui, avec quelles limites et quelles conditions ?**

---

## **Autres Solutions Existantes**

Diverses approches ont été développées dans le but de prédire le prix du pétrole :

* **Modèles économétriques traditionnels** (ARIMA, VAR…) : efficaces à court terme mais peu adaptés à la non-linéarité du marché pétrolier.
* **Approches fondamentales** : s’appuyant sur des indicateurs macroéconomiques (PIB, production, consommation…).
* **Réseaux de neurones** (RNN, LSTM) : exploitent les données temporelles mais peuvent manquer d’interprétabilité.
* **Méthodes d’apprentissage supervisé** (XGBoost, Random Forest…) : fournissent des résultats intéressants mais restent souvent opaques et peu accessibles.

Ces méthodes se concentrent en général uniquement sur la prédiction brute, sans interface utilisateur, sans mise en contexte ni visualisation enrichie.

---

## **Description Détaillée de Notre Solution**

Notre projet propose **une plateforme web interactive** dédiée à la prédiction du prix du pétrole brut (WTI) sur le marché américain. Cette plateforme repose sur l’intégration de l’intelligence artificielle (IA), de la visualisation de données et d’un design intuitif orienté utilisateur. Elle s’adresse aussi bien aux professionnels qu’aux étudiants ou passionnés d’économie énergétique.

### Les fonctionnalités principales :

* **Trois modèles de prédiction complémentaires** :

  * **LSTM** : pour capter les dépendances temporelles.
  * **XGBoost** : performant sur les données tabulaires.
  * **SARIMAX** : pour intégrer la saisonnalité et les variables exogènes.

* **Visualisation claire** des résultats de prédiction : courbes interactives, comparaisons entre modèles, indicateurs de performance (RMSE, MAE, MAPE…).

* **Exploration dynamique** des variables explicatives : production, consommation, PIB, importations/exportations, etc., collectées automatiquement via **web scraping**, avec synthèse textuelle de leur impact.

* **Visualisation de l’historique** des prix du pétrole : courbes interactives retraçant les grandes tendances.

* **Intégration d’actualités économiques** en temps réel à partir de sources fiables (OPEP, marchés financiers, géopolitique...).

* **Accès à la base de données** complète utilisée pour l’entraînement, avec exportation possible en **Excel ou PDF enrichi**.

Cette solution met en avant **la transparence, l’interprétabilité, la pédagogie et l’accessibilité** pour faire de l’intelligence artificielle un outil compréhensible et utile à la prise de décision.

---

## **Conclusion**

L’analyse du marché pétrolier, en raison de sa complexité structurelle et de son importance stratégique, constitue un véritable défi. Nous avons présenté dans ce chapitre les enjeux liés à la volatilité des prix du pétrole, les difficultés rencontrées par les approches classiques, et notre réponse innovante à travers une plateforme intégrée, intelligente et accessible.

Le chapitre suivant décrira la **méthodologie de travail**, les **outils utilisés**, ainsi que **la conception technique** et **UML** de notre solution.

---

# **Chapitre 2 : Méthodologie, Conception UML et Technologies Utilisées**

## **Méthodologie de Travail**

Pour mener à bien ce projet, nous avons adopté une méthode de travail **collaborative et itérative**, s’étalant de **février 2025 à mai 2025**. Le projet a été mené par **trois membres**, chacune ayant un rôle précis dans les différentes phases.

### **1.1 Recherche et exploration du domaine**

Nous avons commencé par une **étude documentaire approfondie** du marché pétrolier. Cette étape a permis de comprendre les **facteurs influençant les prix** et de bâtir les fondations nécessaires pour la modélisation.

### **1.2 Constitution du dataset**

Chaque membre s’est chargée de **collecter des données spécifiques** sur deux facteurs clés (ex. : production, importations, PIB…). Ces données ont été rassemblées pour créer un **jeu de données multidimensionnel**, riche et varié.

### **1.3 Prétraitement et nettoyage**

Chaque membre a nettoyé ses données respectives : suppression des anomalies, interpolation des données manquantes, mise au format des dates, et ordonnancement chronologique. Cette étape a permis d’obtenir un dataset **propre et prêt à l’analyse**.

### **1.4 Construction et évaluation des modèles**

Nous avons implémenté et évalué **trois modèles** :

* **LSTM**
* **XGBoost**
* **SARIMAX**

Chaque modèle a été testé, validé et comparé selon des métriques classiques : **RMSE, MAE, MAPE**. Les meilleures versions ont été sélectionnées et intégrées à la plateforme.

### **1.5 Conception technique**

Nous avons défini ensemble :

* L’**architecture fonctionnelle** de la plateforme.
* Les **technologies utilisées** (Python, Flask, HTML/CSS, Chart.js, Pandas, Scikit-learn, TensorFlow…).
* Les **règles de communication** entre le back-end et le front-end.

### **1.6 Développement des modules**

Le développement a été réparti comme suit :

* **Page d’accueil + navigation** : membre 1.
* **Intégration des modèles + affichage des prédictions** : membre 2.
* **Scraping + visualisation des variables explicatives** : membre 3.

Ensemble, nous avons ensuite implémenté les **fonctionnalités secondaires** :

* Export PDF/Excel
* Affichage des actualités
* Section historique du prix
* Interface utilisateur finale

---
# **Chapitre 3 : Etude et Analyse des Données**
---

## **1. Collection et Compréhension des Données**

### **1.1. Choix du Dataset et Justification**

Dans le cadre de notre projet de prédiction du prix du pétrole, nous avons élaboré un jeu de données multivarié rassemblant plusieurs indicateurs économiques, énergétiques et événementiels. Ce dataset couvre une période étendue sur plusieurs années, avec une granularité mensuelle.

Les variables sélectionnées sont les suivantes :

| **Colonne**   | **Description**                                                                    |
| ------------- | ---------------------------------------------------------------------------------- |
| Date          | Mois et année des observations.                                                    |
| Prix          | Prix moyen mensuel du baril de pétrole (en USD).                                   |
| Import        | Volume mensuel de pétrole importé (en milliers de barils).                         |
| Export        | Volume mensuel de pétrole exporté.                                                 |
| Production    | Production pétrolière mensuelle aux États-Unis (en milliers de barils/jour).       |
| Inflation (%) | Taux d'inflation mensuel aux États-Unis.                                           |
| GDP           | Produit Intérieur Brut mensuel ou trimestriel (en milliards de dollars).           |
| Event         | Événements majeurs pouvant impacter le marché pétrolier (catégorisation manuelle). |

---

### **1.2. Raisons du Choix des Variables**

Ces variables ont été choisies en raison de leur influence directe ou indirecte sur les fluctuations du prix du pétrole :

* **Importations, exportations et production** : Reflètent la dynamique de l’offre et de la demande sur le marché intérieur et international.
* **Inflation** : Indicateur de la pression économique globale qui peut affecter la consommation d’énergie.
* **GDP (Produit Intérieur Brut)** : Représente l'activité économique globale, en lien étroit avec la demande énergétique.
* **Événements** : Les événements externes (géopolitiques, économiques ou naturels) ont souvent un impact significatif et immédiat sur les prix du pétrole.

---

### **1.3. Catégorisation des Événements**

La variable **Event** a été classifiée manuellement en quatre grandes catégories d'événements influençant potentiellement le marché pétrolier :

* **Global Economic Events** : Crises financières, récessions, krachs boursiers.
* **Oil Production & Policy** : Décisions de l’OPEP, modifications des quotas, embargos.
* **Geopolitical Events** : Conflits armés, sanctions internationales, tensions diplomatiques.
* **Natural Disasters** : Ouragans, séismes, tempêtes affectant les infrastructures de production.

---

### **1.4. Analyse Préliminaire de l'Influence des Variables**

Des analyses statistiques et visuelles ont été réalisées pour évaluer l’influence de chaque variable sur le prix du pétrole :

* **Distribution du prix selon les événements**
  ➤ Les catastrophes naturelles génèrent en moyenne les prix les plus élevés.

* **Corrélation glissante (6 mois) entre production et prix**
  ➤ Illustre la variabilité temporelle de la corrélation, qui n’est pas constante.

* **Matrice de corrélation**
  ➤ Montre les liens linéaires entre les variables.
  → Le **GDP** est le plus fortement corrélé au prix du pétrole (**corrélation de 0.87**).
  → Les **importations et exportations** présentent une corrélation modérée.
  → L’**inflation** semble avoir peu d’impact direct.

* **Prix moyen selon type d’événement**
  ➤ Les catastrophes naturelles et les crises économiques mondiales sont associées aux niveaux de prix les plus élevés.

* **Analyse bivariée avec graphiques** :
  → **Prix vs Export** : Corrélation faible et diffuse.
  → **Prix vs GDP** : Corrélation positive claire.
  → **Prix vs Import** : Relation modérée.
  → **Prix vs Inflation** : Corrélation très faible.
  → **Prix vs Production** : Relation complexe, potentiellement inversée mais instable.

---

## **2. Prétraitement des Données**

### **2.1. Nettoyage des Données**

Un nettoyage rigoureux est indispensable pour assurer la robustesse et la fiabilité du modèle prédictif. Cette étape a principalement porté sur le traitement des valeurs manquantes et aberrantes.

#### **a) Traitement des valeurs manquantes**

Pour préserver la cohérence temporelle des séries, nous avons utilisé l’**interpolation linéaire** afin d’estimer les données manquantes :

* **Colonne "Prix" (1930–1973)** : Les données n'étaient disponibles qu’annuellement. Une interpolation mensuelle a été appliquée.
* **Colonne "Export"** : Contenait des lacunes sur plusieurs périodes. Une interpolation linéaire a été utilisée pour reconstituer les données.
* **Colonne "GDP"** : Données trimestrielles uniquement. Une interpolation linéaire mensuelle a été appliquée pour une meilleure granularité.

#### **b) Gestion des valeurs aberrantes**

Les valeurs aberrantes peuvent perturber l'apprentissage des modèles, en particulier pour les réseaux de neurones LSTM. Deux méthodes ont été testées :

* **IQR (Interquartile Range)** : Méthode statistique simple pour détecter les valeurs extrêmes dans des distributions univariées.
* **Isolation Forest** : Méthode non supervisée, efficace pour détecter des anomalies multivariées.

**Choix final** : Plutôt que de supprimer ou modifier les outliers, nous avons opté pour une **normalisation** des données afin de préserver l’intégrité temporelle :

* **Méthodes utilisées** : Min-Max Scaling ou Z-score.
* **Avantages** :

  * Réduction de l’influence des valeurs extrêmes.
  * Conservation de la structure séquentielle des données.
  * Uniformisation des échelles pour l’ensemble des variables.

---

### **2.2. Transformation des Données**

Avant l'entraînement du modèle LSTM, plusieurs transformations ont été appliquées :

#### **a) Encodage des variables catégorielles**

La variable **"Event"** a été encodée par **One-Hot Encoding** via `pandas.get_dummies()` et `OneHotEncoder()` de Scikit-learn. Cette méthode permet :

* D’intégrer des catégories qualitatives sans hiérarchie artificielle.
* De rendre les événements exploitables par des modèles d’apprentissage.

#### **b) Normalisation des variables numériques**

Les variables numériques (prix, importations, exportations, GDP, etc.) ont été normalisées pour :

* Faciliter la convergence du modèle.
* Garantir une homogénéité entre les différentes échelles.
* Réduire l’impact des valeurs extrêmes sans les supprimer.

---



---

## 📈 Modélisation des prix du pétrole

Ce projet explore différentes approches de modélisation pour prédire les prix du pétrole à partir de données économiques historiques et événementielles. Trois méthodes principales ont été utilisées : **LSTM**, **SARIMAX/ARIMAX**, et **XGBoost**.

---

### 1️⃣ Modélisation avec LSTM (Long Short-Term Memory)

#### 1.1 Objectif

LSTM est un type de réseau de neurones récurrents conçu pour modéliser des séquences longues, en évitant le problème du gradient qui disparaît. Il est utilisé ici pour prédire les prix du pétrole à partir d’un historique de 12 mois, enrichi de variables telles que l’inflation, le PIB, les importations/exportations, la production, et certains événements.

#### 1.2 Principe de fonctionnement

Un bloc LSTM contient trois types de portes :

* **Porte d’entrée** : sélectionne les nouvelles informations à stocker.
* **Porte d’oubli** : décide quelles anciennes informations seront supprimées.
* **Porte de sortie** : génère la sortie pour l'étape suivante.

#### 1.3 Préparation des données

* **Encodage** des variables catégorielles (événements) avec OneHotEncoder.
* **Normalisation** des données avec MinMaxScaler.
* **Transformation séquentielle** : les données sont converties en séquences de 12 mois pour chaque entrée du modèle.

#### 1.4 Architecture du modèle

```python
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dense(1)
])
```

| Composant                | Description                                 |
| ------------------------ | ------------------------------------------- |
| `Sequential()`           | Modèle séquentiel empilant les couches      |
| `LSTM(64)`               | Couche LSTM avec 64 neurones                |
| `return_sequences=False` | Retourne uniquement la dernière sortie      |
| `Dense(1)`               | Neurone de sortie pour la prédiction finale |

#### 1.5 Paramètres d'entraînement

| Paramètre         | Valeur | Description                |
| ----------------- | ------ | -------------------------- |
| optimizer         | adam   | Optimiseur adaptatif       |
| loss              | mse    | Erreur quadratique moyenne |
| epochs            | 100    | Nombre d’époques           |
| batch\_size       | 16     | Taille du batch            |
| validation\_split | 0.1    | Données de validation      |

#### 1.6 Évaluation

* Les prédictions sont inversées (post-normalisation).
* Métriques :

  * **MAE** : Erreur absolue moyenne
  * **RMSE** : Erreur quadratique moyenne
  * **Précision personnalisée** : `100 - (MAE / Moyenne réelle) * 100`
* Une interprétation du niveau de précision est générée automatiquement.

#### 1.7 Sauvegarde

| Élément             | Fichier                              |
| ------------------- | ------------------------------------ |
| Modèle LSTM         | `lstm_model.h5`                      |
| Scaler              | `scaler_lstm.pkl`                    |
| Index cible         | `target_index.pkl`                   |
| Données normalisées | `scaled_data.pkl`                    |
| Données de test     | `x_test_lstm.npy`, `y_test_lstm.csv` |

#### ✅ Synthèse

Le modèle LSTM s’est montré pertinent pour capter les dépendances temporelles du prix du pétrole. Les pistes d’amélioration incluent :

* Ajouter davantage de données
* Introduire du dropout ou des couches supplémentaires
* Ajuster les hyperparamètres

---

### 2️⃣ Modélisation avec ARIMAX / SARIMAX

#### 2.1 Objectif

ARIMAX est une extension d’ARIMA qui intègre des variables explicatives (exogènes). L’objectif est de modéliser le prix du pétrole en tenant compte du PIB, des importations/exportations, de la production, etc.

#### 2.2 Pourquoi ARIMAX ?

Ce modèle permet :

* De capter les tendances internes à la série (ARIMA)
* D'intégrer des variables économiques influentes (exogènes)
* D'améliorer la précision par rapport à ARIMA seul

#### 2.3 Fonctionnement

Le modèle intègre :

* **AR (AutoRegressive)** : dépendance aux valeurs passées
* **I (Integrated)** : différenciation pour la stationnarité
* **MA (Moving Average)** : erreurs passées
* **X (eXogène)** : variables explicatives

#### 2.4 Préparation des données

* Division en ensemble d'entraînement/test (80/20)
* Test de stationnarité (ADF) + différenciation (d=1)
* Standardisation des variables exogènes
* Synchronisation temporelle des variables

#### 2.5 Architecture

Utilisation de `auto_arima` pour déterminer les paramètres optimaux (p,d,q), puis modélisation avec **SARIMAX** de `statsmodels`.

#### 2.6 Paramètres

| Élément             | Valeur                                            |
| ------------------- | ------------------------------------------------- |
| d (différenciation) | 1                                                 |
| p, q                | déterminés automatiquement                        |
| Variables exogènes  | PIB, Import, Export, Production, Inflation, Event |

#### 2.7 Évaluation

| Métrique | Résultat |
| -------- | -------- |
| RMSE     | 27.45    |
| MAE      | 22.22    |
| R²       | -0.546   |

> ⚠️ Le modèle n’a pas su capter les dynamiques récentes du marché malgré les variables explicatives.

#### ❌ Synthèse

Le modèle ARIMAX s’est révélé limité :

* Variables exogènes insuffisantes
* Comportement du marché probablement non-linéaire
* Sensibilité à la stationnarité et aux paramètres

💡 Recommandation : envisager des approches non-linéaires comme les réseaux neuronaux ou les modèles hybrides.

---

### 3️⃣ Modélisation avec XGBoost

#### 3.1 Définition

XGBoost est un algorithme de boosting d’arbres de décision qui construit les arbres de manière séquentielle pour corriger les erreurs précédentes. Il est réputé pour sa vitesse, sa robustesse et sa précision.

#### 3.2 Fonctionnement

* Chaque nouvel arbre corrige les erreurs du précédent.
* Optimisation de la fonction de perte par gradient.
* Intégration de la régularisation pour limiter le surapprentissage.

#### 3.3 Pourquoi XGBoost ?

* Capture bien les relations non linéaires
* Robuste face aux données manquantes
* Gère les effets combinés (ex. : production + export)
* Efficace sur de gros volumes de données

#### 3.4 Préparation des données

* Nettoyage et formatage
* Création de variables temporelles (mois, trimestre, etc.)
* Normalisation ou standardisation si nécessaire
* Intégration de toutes les variables pertinentes : PIB, importations, exportations, inflation, événements, etc.

> (🔧 La section complète sur l’architecture, l'entraînement et l’évaluation du XGBoost est à compléter ici si besoin.)

---

## 📌 Conclusion générale

Chaque approche présente des avantages :

* **LSTM** pour la modélisation de séquences non-linéaires
* **ARIMAX** pour une approche classique interprétable intégrant des variables économiques
* **XGBoost** pour sa flexibilité, sa robustesse et sa capacité à capter des patterns complexes

🔄 Des approches hybrides ou une optimisation approfondie peuvent améliorer davantage les performances des modèles prédictifs.

---





