# 🎬 y_BoxOffice: Prédiction de Revenus Cinématographiques

![Banner](https://via.placeholder.com/1200x300/4a5568/ffffff?text=y_BoxOffice+%7C+Pr%C3%A9diction+de+Revenus+Cin%C3%A9matographiques)

## 📊 Vue d'ensemble du projet

> *"Prédire le succès, c'est commencer à le créer."*

Ce projet utilise l'apprentissage automatique pour prédire les revenus au box-office des films en analysant divers facteurs qui influencent le succès commercial d'un film, des genres et dates de sortie aux sociétés de production et au contenu.

---

## 💡 Motivation

L'industrie cinématographique investit des milliards dans la production et le marketing, rendant la prédiction précise des revenus cruciale pour les studios et les investisseurs. Ce modèle aide à comprendre ce qui détermine la performance au box-office et fournit un outil pour estimer les résultats financiers des sorties à venir.

---

## 🗃️ Données

Notre ensemble de données contient des informations sur plus de 13 000 films, notamment:

| Type de données | Description |
|----------------|-------------|
| 📋 Métadonnées | Titre, durée, date de sortie |
| 🎭 Genres | Classifications par genre |
| 🏢 Production | Sociétés de production |
| 📝 Contenu | Descriptions/synopsis des films |
| ⭐ Évaluations | Notes des utilisateurs et métriques de popularité |

---

## 🔬 Méthodologie

```mermaid
graph LR
    A[Données Brutes] --> B[Nettoyage]
    B --> C[Ingénierie des Caractéristiques]
    C --> D[Entraînement du Modèle]
    D --> E[Évaluation]
    E --> F[Sélection du Modèle]
    F --> G[Prédiction des Revenus]
    
    style A fill:#d0e0ff,stroke:#333
    style G fill:#c0ffc0,stroke:#333
```

1. **🧹 Nettoyage des données**: Suppression des anomalies, des films expérimentaux et des entrées avec des valeurs irréalistes
2. **🔍 Analyse exploratoire**: Analyse de la distribution des revenus et des corrélations clés
3. **⚙️ Ingénierie des caractéristiques**: Extraction de caractéristiques temporelles, d'indicateurs de genre et d'insights textuels
4. **🤖 Développement du modèle**: Implémentation de plusieurs algorithmes ML et comparaison des performances
5. **📏 Évaluation**: Évaluation de la précision prédictive à l'aide de métriques pertinentes pour l'industrie

---

## 🌟 Caractéristiques clés

| Catégorie | Caractéristiques |
|-----------|-----------------|
| ⏱️ **Temporelles** | Mois, année et saison de sortie |
| 📋 **Contenu** | Classifications par genre, durée |
| 🏢 **Production** | Implication du studio, indicateur de grand studio |
| 📝 **Texte** | Caractéristiques NLP extraites des descriptions |
| 👥 **Popularité** | Nombre de votes, métriques de réception du public |

---

## 🧠 Modèles

Nous comparons plusieurs algorithmes:

> 🌲 **Random Forest Regression**  
> 📈 **Gradient Boosting Machines**  
> 📊 **Régression linéaire** (référence)

---

## 💎 Insights

Notre analyse a révélé:

- Les sorties estivales et pendant les vacances performent généralement mieux
- L'Aventure et la Science-Fiction sont les genres les plus lucratifs
- Les films populaires (nombre élevé de votes) sont fortement corrélés aux revenus
- La qualité des évaluations du public montre une corrélation minimale avec le succès financier
- Le soutien d'un grand studio impacte significativement les revenus

---

## ⚙️ Installation et utilisation

```bash
# Cloner le dépôt
git clone https://github.com/votrenomdutilisateur/y_BoxOffice.git

# Installer les dépendances
pip install -r requirements.txt

# Exécuter les notebooks
jupyter notebook
```

---

## 📋 Prérequis

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- NLTK ou spaCy (pour les fonctionnalités NLP)

---

<div align="center">
<p><i>Projet développé dans le cadre du programme Albert Global Data School 2024-2025</i></p>
</div>