#!/bin/bash

# Run by devcontainer's 'onCreateCommand'
#
# Gets and install Spark 3.5.5 in /opt/spark. Also installs the system default
# Java development kit via the package manager

sudo apt-get update
sudo apt -y upgrade
sudo apt-get install -y default-jdk
curl https://dlcdn.apache.org/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz --output /tmp/spark.tgz
sudo mkdir /opt/spark
sudo tar -xf /tmp/spark.tgz -C /opt/spark --strip-components 1