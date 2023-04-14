#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

master_node_ip = '10.10.1.1'

client_nodes_addr = {
    1:'10.10.1.2',2:'10.10.1.3', 3:'10.10.1.4',4:'10.10.1.5',5:'10.10.1.6',
    6:'10.10.1.7',7:'10.10.1.8',8:'10.10.1.9',9:'10.10.1.10',10:'10.10.1.11'
    
}

#mongodb_url = 'mongodb://jahanxb:phdunr@130.127.133.239:27017/?authMechanism=DEFAULT&authSource=flmongo&tls=false'
#mongodb_url = 'mongodb+srv://jahanxb:phdunr@flmongo.7repipw.mongodb.net/?retryWrites=true&w=majority'

# mongodb_url = 'mongodb://jahanxb:phdunr@10.10.1.1:27017/?authMechanism=DEFAULT&authSource=flmongo&tls=false'

#mongodb_url = 'mongodb://jahanxb:phdunr@10.10.1.1:27017/iteration_status?authMechanism=DEFAULT&authSource=admin&tls=false'

mongodb_url = 'mongodb://jahanxb:phdunr@10.10.1.1:27017/?authMechanism=DEFAULT&authSource=admin&tls=false'

cassandra_addr = '10.10.1.2'