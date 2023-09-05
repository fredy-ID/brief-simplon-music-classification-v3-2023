# brief-simplon-music-classification-v3-2023
 Création d'une page de test pour les modèles de classification de musique

### 🛠️Créer l'environment virtuel
Ce dernier ira contenir toutes les dépendances [*(Bibliothèques)*](#dépendances) de votre backend, n'oubliez pas de le référencer dans le [.gitignore](https://www.git-scm.com/docs/gitignore), de même pour la base de donnée et le fichier `secret.py`, lui même dans le dossier [src](./backend/src/)
```sh
python -m venv ./venv
```

### 🟢Activer l'environment virtuel
Avec powershell
```sh
.\venv\Scripts\activate
```
Avec CMD
```sh
venv\Scripts\activate.bat
```

#### 🔴Pour le désactiver, si besoin utiliser la commande suivante
```sh
deactivate
```

### Installer les bonnes [dépendances](#dépendances)
```sh
cd backend
pip install -r requirements.txt
pip install joblib pandas IPython librosa matplotlib
```

### Lancer le serveur django
Avec powershell
```sh
py manage.py runserver
```


### 📚Arborescence
⚠️ Vision simplifié de l'arborescence
<details>
<summary>BRIEF-SIMPLON-MUSIC-CLASSIFICATION-V3-2023</summary>

- 📂backend (Dossier de travail)
  - 📁src
  - 💾db.qlite3
  - </>manage.py
</details>