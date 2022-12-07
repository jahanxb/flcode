#!/bin/bash

sudo systemctl stop cassandra


#sudo rm -rf /var/lib/cassandra/data/system/*
sudo rm -rf /var/lib/cassandra/*


sudo systemctl start cassandra

