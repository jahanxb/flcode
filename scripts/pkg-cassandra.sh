#!/bin/bash

sudo apt-get update -y

sudo apt install default-jdk -y

sudo apt install wget -y

wget -q -O - https://www.apache.org/dist/cassandra/KEYS | sudo apt-key add -

echo "deb http://www.apache.org/dist/cassandra/debian 40x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list deb http://www.apache.org/dist/cassandra/debian 40x main

sudo apt-get update -y

sudo apt install cassandra -y

sudo systemctl enable cassandra

sudo systemctl start cassandra


