#!/bin/bash
echo "Welcome"

sudo apt-get update

sudo apt-get install ca-certificates -y
sudo apt-get install curl -y
sudo apt-get install gnupg -y
sudo apt-get install lsb-release -y


sudo mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null


sudo apt-get update

sudo apt-get install docker-ce -y
sudo apt-get install docker-ce-cli -y
sudo apt-get install containerd.io -y
sudo apt-get install docker-compose-plugin -y
sudo apt-get install docker-compose -y
