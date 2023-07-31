#!/bin/bash

# Update and upgrade packages
sudo apt-get update && sudo apt-get upgrade

# Install Node.js
sudo apt-get install -y nodejs

# Install Python dependencies and NGINX
sudo apt-get -y install python3-pip python3-venv nginx

# Create Python virtual environment and activate it
python3 -m venv venv
. venv/bin/activate

# Install npm and pm2
sudo apt-get install -y npm
sudo npm install -g pm2

# clone
git clone -b mobile_server https://github.com/setiawanboedy/uno.git

# Install library python
pip install -r requierements.txt

# Start the Gunicorn server with UVicorn worker using pm2
pm2 start "gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app" --name uno

# donwload file conf
wget -O "/etc/nginx/conf.d/default.conf" "https://raw.githubusercontent.com/setiawanboedy/uno/mobile_server/default.conf"

# Restart NGINX to apply the changes
sudo service nginx restart