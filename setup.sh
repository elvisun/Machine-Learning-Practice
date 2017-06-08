#!/bin/bash

sudo apt-get update
sudo apt-get install python-setuptools python-dev build-essential 
sudo easy_install pip 

# get NVDIA support
sudo apt-get install libcupti-dev

sudo pip install -r requirements.txt
