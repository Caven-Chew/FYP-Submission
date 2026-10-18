# BrewBot + RoboBot, the SC3098 practice chatbot. Built and started by compose.yaml in this folder.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 BREWBOT_HOST=0.0.0.0 BREWBOT_PORT=5056
WORKDIR /brewbot
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5056
CMD ["python", "bot.py"]
