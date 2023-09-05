# Utilisez une image de base Node.js
FROM node:14

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers package.json et package-lock.json pour installer les dépendances
COPY package*.json ./

# Installez les dépendances
RUN npm install

# Copiez le reste des fichiers de l'application
COPY . .

# Exposez le port 3000 pour le serveur de développement SolidJS
EXPOSE 3000

# Démarrez l'application SolidJS en mode développement
CMD ["npm", "start"]
