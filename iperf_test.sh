#!/bin/bash

# Get the IP address of the node
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Node IP: $NODE_IP"

# Base port number
BASE_PORT=5201

# Function to run iperf3 server
run_server() {
    local port=$1
    echo "Starting iperf3 server on $NODE_IP:$port"
    iperf3 -s -B $NODE_IP -p $port &
    echo $!
}

# Function to run iperf3 client
run_client() {
    local port=$1
    local gpu_num=$2
    echo "Testing GPU $gpu_num to GPU 0"
    iperf3 -c $NODE_IP -p $port -t 30 -f m
}

# Run tests for each GPU pair
for i in {1..3}; do
    port=$((BASE_PORT + i))
    
    # Start server
    server_pid=$(run_server $port)
    
    # Wait for server to start
    sleep 5
    
    # Run client
    run_client $port $i
    
    # Stop server
    kill $server_pid
    
    echo ""
done

echo "All tests completed."