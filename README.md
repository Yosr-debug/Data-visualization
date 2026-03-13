# Cardiovascular Risk Explorer

Application Streamlit réalisée pour le **Parcours B - Projet personnel**.

## Objectif
Explorer un dataset cardiovasculaire et répondre à la question suivante :

**Quels facteurs semblent les plus associés à une hausse du risque cardiovasculaire ?**

## Contenu de l'application
L'application contient 4 sections :

1. **Accueil**
   - présentation du projet
   - contexte, objectifs, source des données
   - KPIs globaux
   - aperçu des données
   - description des colonnes

2. **Exploration et visualisations**
   - 6 KPIs
   - 6 visualisations interactives
   - filtres sur l'âge, la catégorie de risque, le tabagisme et les antécédents familiaux

3. **Analyse approfondie**
   - question de recherche
   - 4 graphiques analytiques
   - insights et limites

4. **Dashboard interactif**
   - vue synthétique
   - graphiques combinés
   - téléchargement des données filtrées

## Lancer le projet en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement Streamlit Cloud
1. Créer un dépôt GitHub
2. Ajouter les fichiers du projet
3. Pousser le dépôt sur GitHub
4. Aller sur Streamlit Cloud
5. Choisir le dépôt
6. Sélectionner `app.py` comme fichier principal
7. Déployer

## Fichiers
- `app.py` : application principale
- `cardiovascular_risk_dataset.csv` : dataset
- `requirements.txt` : dépendances
- `README.md` : documentation
