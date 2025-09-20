# Install stage
FROM node:alpine AS install
WORKDIR /app
COPY package.json .
RUN npm install
COPY *.ts *.html *.json ./
COPY src/ src/
COPY public/ public/

# Development image
FROM install AS dev
WORKDIR /app
COPY .env.development .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]

# Build stage
FROM install AS build
WORKDIR /app
# Build time arguments
ARG API_URL="/api/v1"
# Set run time variables for react app from build arguments
ENV VITE_API_URL=$API_URL
RUN npm run build

# Production image
FROM nginx:latest AS prod
COPY --from=build /app/dist /usr/share/nginx/html