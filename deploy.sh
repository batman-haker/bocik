#!/bin/bash
# Deploy arbitrage bot to mikr.us server
# Usage: ssh user@srv.mikr.us -p PORT < deploy.sh
#    or: scp this to server and run there

set -e

echo "=== Funding Rate Arbitrage Bot — mikr.us deploy ==="

# 1. Install Docker if needed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Installing docker-compose..."
    apt-get update && apt-get install -y docker-compose-plugin
fi

# 2. Clone/pull repo
REPO_DIR="$HOME/arb-bot"
if [ -d "$REPO_DIR" ]; then
    echo "Pulling latest..."
    cd "$REPO_DIR" && git pull
else
    echo "Cloning repo..."
    git clone https://github.com/YOUR_USERNAME/funding-rate-arb.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

# 3. Set port (mikr.us assigns random IPv4 port)
if [ -z "$PORT" ]; then
    echo ""
    echo "IMPORTANT: mikr.us uses random IPv4 ports!"
    echo "Check your port in the mikr.us panel."
    echo "Set it with: export PORT=<your_port>"
    echo ""
    read -p "Enter your mikr.us port: " PORT
fi

export PORT
echo "PORT=$PORT" > .env

# 4. Build and run
echo "Building Docker image..."
docker compose build

echo "Starting bot..."
docker compose up -d

echo ""
echo "=== Bot deployed! ==="
echo "Dashboard: http://your-mikrus-ip:$PORT"
echo ""
echo "Useful commands:"
echo "  docker compose logs -f        # follow logs"
echo "  docker compose restart         # restart bot"
echo "  docker compose down            # stop bot"
echo "  docker compose up -d --build   # rebuild & restart"
