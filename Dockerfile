FROM python:3.11-slim

# Working dir is PARENT of arbitrage_system (for absolute imports)
WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt /app/arbitrage_system/
RUN pip install --no-cache-dir -r /app/arbitrage_system/requirements.txt

# Copy source code
COPY . /app/arbitrage_system/

# Create data directories
RUN mkdir -p /app/arbitrage_system/logs \
             /app/arbitrage_system/reports \
             /app/arbitrage_system/shadow_trades \
             /app/arbitrage_system/backtest_results

# Default port (override with PORT env on mikr.us)
ENV PORT=8080
ENV NO_BROWSER=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "arbitrage_system/bot.py"]
