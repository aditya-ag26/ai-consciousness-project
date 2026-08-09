# Frontend image: builds the React bundle, then serves it as static files.
#
# Vite inlines environment variables at build time, so the API URL is a build
# argument rather than a runtime one. Point it at the deployed backend when
# building for production.

# ---------- build stage ----------
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies from the lockfile first so this layer is reused whenever
# only application source has changed.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# ---------- runtime stage ----------
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --from=builder /app/dist /usr/share/nginx/html

# Cloud Run assigns the port at runtime. The nginx image substitutes variables
# in templates/ at startup, so PORT reaches the served config.
ENV PORT=8080
EXPOSE 8080
