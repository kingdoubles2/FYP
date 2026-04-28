FROM node:22-alpine

WORKDIR /app

COPY src/frontend/package.json /app/package.json
RUN npm install

COPY src/frontend /app

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
