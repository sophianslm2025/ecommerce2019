# README — daily_quantity_forecast.py

Ce script Python permet de prédire la quantité quotidienne de commandes pour l’e‑commerce 2019, en testant plusieurs modèles (SARIMA, Holt‑Winters, LSTM) et en exportant les prévisions sur 30 jours.

---
## 🛠️ Prérequis

1. **Python 3.8+** installé (recommandé 3.10 ou 3.11).
2. Un gestionnaire de dépendances (pip).
3. Base MySQL `ecommerce_2019` accessible avec les données nettoyées (table `orders_2019`).

---
## 📂 Contenu du projet

```text
loan_dashboard/
└─ backend/
   ├ daily_quantity_forecast.py   # Script principal
   ├ requirements.txt             # Dépendances Python
   └─ model/                      # (optionnel) modèles sauvegardés
      └ pipeline.pkl              # Exemple ou modèle final
```

---
## 📥 Installation

1. **Cloner ou copier** ce dossier `backend`.
2. Dans un terminal, **créer et activer** un environnement virtuel :

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Installer** les dépendances :

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

---
## 🔧 Configuration

- **DB_CONFIG** : ouvrir `daily_quantity_forecast.py`, et adapter la section :

  ```python
  DB_CONFIG = {
      "user": "root",          # utilisateur MySQL
      "password": "",          # mot de passe
      "host": "localhost",     # hôte MySQL
      "port": 3306,              # port
      "database": "ecommerce_2019",
  }
  ```

- **pipeline.pkl** : placer, si existant, votre pipeline entraîné dans `backend/model/pipeline.pkl`. Sinon, laissez‑le vide pour générer un stub.

---
## ▶️ Exécution

1. **Lancer** le script :

   ```bash
   python daily_quantity_forecast.py
   ```

2. **Sorties** produites :
   - `serie_journaliere.png` : graphique de la série journalière historique.
   - `daily_quantity_forecast.csv` : CSV des prévisions sur 30 jours.
   - `best_model_<type>.pkl` ou `best_model_lstm.h5` : fichier du modèle sélectionné.

---
## 📊 Structure du script

1. **Connexion MySQL** via SQLAlchemy et extraction des quantités journalières.
2. **Analyse rapide** : trace la série historique.
3. **Split train/test** sur dernier mois.
4. **Entraînement** de trois modèles :
   - SARIMA
   - Holt‑Winters
   - LSTM
5. **Évaluation** (MAE, RMSE) et sélection automatique.
6. **Prévisions** sur 30 jours et export des résultats.

---
## 📝 Personnalisation

- **Horizons de prévision** : modifier `forecast_horizon`.
- **Paramètres SARIMA et LSTM** : ajuster `order`, `seasonal_order`, `look_back`, etc.
- **Enregistrer** d’autres métriques ou visualisations en ajoutant du code après la partie sélection.

---
## ⚠️ Remarques

- Assurez-vous que la table `orders_2019` contient bien les données jusqu’à la date maximale souhaitée.
- Pour un modèle de production, envisagez de gérer les exceptions de connexion MySQL et de proposer un fallback.

---
## 📚 Références

- `statsmodels` pour SARIMA et Holt‑Winters
- `tensorflow.keras` pour LSTM
- `pandas` et `SQLAlchemy` pour la manipulation et extraction des données

---