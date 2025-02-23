#!/bin/bash

# Hosts
HOST1="yanPC2@172.18.156.204"
HOST2="yanPC3@172.18.156.77"
HOST3="yanPC4@172.18.156.251"

gnome-terminal --window --title="yanPC3" -- bash -c "ssh $HOST2; exec bash"
gnome-terminal --window --title="yanPC2" -- bash -c "ssh $HOST1; exec bash"
gnome-terminal --window --title="yanPC4" -- bash -c "ssh $HOST3; exec bash"

echo "Launched terminals and connected to hosts."
