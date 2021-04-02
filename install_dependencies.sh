#!/bin/bash
pip install -r requirements.txt
apt update && apt install -y default-jdk
apt install bc
cd /media/bizon/DATA/guillaume/patent_database/src/
bash install.sh
cd /media/bizon/DATA/guillaume/bert_summary/src/
apt install vim -y
