# SERVER SIDE
def FedAvg_Server(K, T, C, E, B, η):
    """
    Parameters:
    - K: Total number of clients
    - T: Number of communication rounds
    - C: Fraction of clients selected each round (e.g., 0.6)
    - E: Number of local epochs per client
    - B: Local batch size
    - η: Learning rate
    """
    
    # Initialize global model
    w_global = initialize_model()
    
    for round t = 1, 2, ..., T:
        # 1. Select random subset of clients
        m = max(C * K, 1)  # Number of clients to select
        S_t = random_sample(clients, m)  # Selected clients
        
        # 2. Broadcast global model to selected clients
        for each client k in S_t:
            send(w_global, to=client_k)
        
        # 3. Receive model updates from clients
        updates = []
        for each client k in S_t:
            w_k = receive_update_from(client_k)
            n_k = get_data_size(client_k)
            updates.append((w_k, n_k))
        
        # 4. Aggregate updates (Weighted Average)
        w_global = aggregate(updates)
        
    return w_global