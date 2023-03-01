from pymongo import MongoClient
from neo4j import GraphDatabase
from psycopg2 import connect
import pika
from cassandra.cluster import Cluster
import time

def measure_communication_cost_global_model(num_clients, num_iterations, model_size):
    """
    Measures the communication cost on the global model in a federated learning system.
    
    Args:
    num_clients (int): The number of clients in the system.
    num_iterations (int): The number of iterations performed in the system.
    model_size (int): The size of the global model in bytes.
    
    Returns:
    communication_cost (float): The total communication cost in MB.
    """
    # Assuming a fixed number of updates per iteration
    updates_per_iteration = 10
    
    # Assuming a fixed update size per client per iteration
    update_size = 3200
    
    # Calculate the total communication cost
    communication_cost = num_clients * num_iterations * updates_per_iteration * update_size / (1024 * 1024)
    
    # Add the cost of sending the global model to each client at the beginning of each iteration
    communication_cost += num_clients * model_size / (1024 * 1024)
    
    return communication_cost


if __name__ == "__main__":
    
    comm = measure_communication_cost_global_model(num_clients=10, num_iterations=10, model_size=3200)
    print('communication cost: ' + str(comm))
    
    #3.2MB per iteration, per client for cifar
    #9.6MB per iteration, per client for fmnist 
    #20MB per iteration, per client for svhn