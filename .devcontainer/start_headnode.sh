#!/bin/bash

# Run by devcontainer's 'postAttachCommand'
#
# Places current Codespace's host name in .devcontainer/headnode_hostname and
# pushes to GitHub for workers to read. Starts Spark master listening on 
# localhost port 7077.

# Deal with setting headnode URL on GitHub
git pull
echo $CODESPACE_NAME > .devcontainer/headnode_hostname
git add .devcontainer/headnode_hostname
git commit -m "Updated headnode hostname"
git push origin main

# Make sure PySpark uses the correct python
export PYSPARK_PYTHON='/usr/local/bin/python'
export PYSPARK_DRIVER_PYTHON='/usr/local/bin/python'

# Start a master and worker process
sudo /opt/spark/sbin/start-master.sh --port 7077 --host localhost --webui-port 8080
sudo /opt/spark/sbin/start-worker.sh localhost:7077 --webui-port 8081
