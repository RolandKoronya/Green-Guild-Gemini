# Use the official Node.js image.
# https://hub.docker.com/_/node
FROM node:20-slim

# Create and change to the app directory.
WORKDIR /usr/src/app

# Copy application dependency manifests to the container image.
# A wildcard is used to ensure both package.json AND package-lock.json are copied.
COPY package*.json ./

# Install production dependencies.
RUN npm install --only=production

# Copy local code to the container image.
COPY . .

# Cloud Run sets the PORT environment variable to 8080 by default.
# We expose this port to tell Google we are listening here.
EXPOSE 8080

# Run the web service on container startup.
CMD [ "node", "server.js" ]