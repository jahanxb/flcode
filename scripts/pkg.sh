#!/bin/bash

#installation python3.6
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -y
sudo apt-get install python3.6 -y
sudo apt-get install python3.6-dev
sudo apt-get install python3.6-venv

#docker installation
sudo apt-get update -y
 sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release -y

sudo mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y

sudo apt-get install docker-io -y
sudo apt-get install docker-compose -y
