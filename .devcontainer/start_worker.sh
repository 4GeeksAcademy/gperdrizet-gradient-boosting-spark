#!/bin/bash

# Run by devcontainer's 'postAttachCommand'
#
# Updates headnode hostname from GitHub, creates new SSH configuration file
# and opens SSH tunnel to hdeadnode on port 7077. Then starts Spark worker
# process

# Update the headnode hostname and open SSH tunnel
mkdir -p ~/.ssh
git pull
gh codespace ssh -c `cat .devcontainer/headnode_hostname` --config > ~/.ssh/config
ssh -vfN -L 7077:localhost:7077 cs.`cat .devcontainer/headnode_hostname`.main

# Make sure PySpark uses the correct python
export PYSPARK_PYTHON='/usr/local/bin/python'
export PYSPARK_DRIVER_PYTHON='/usr/local/bin/python'

# Start the worker
sudo /opt/spark/sbin/start-worker.sh localhost:7077 --webui-port 8080