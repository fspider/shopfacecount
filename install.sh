#!/bin/bash
sudo apt-get install python3-pip python3-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
sudo apt-get install libqtgui4
sudo apt-get install python3-pyqt5
sudo apt-get install libqt4-test

pip3 install opencv-python
pip3 install face_recognition

sudo apt-get install python-h5py
pip3 install tensorflow
PATH=$PATH:home/pi/.local/bin

# install PostgreSql  
pip3 install psycopg2
