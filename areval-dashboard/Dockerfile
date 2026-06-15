# AREval Dashboard Dockerfile
# Next.js 15 static export — single stage, only pulls node image

FROM node:22-alpine
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

ARG NEXT_PUBLIC_API_URL=http://localhost:8700
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

COPY . .
RUN npm run build

# Serve static files with Node http-server (lightweight, no nginx needed)
RUN npm install -g http-server
EXPOSE 3000
CMD ["http-server", "dist", "-p", "3000"]
