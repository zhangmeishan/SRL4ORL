nohup /data/disk1/zrr/anaconda3/bin/python -u  driver/Train.py --config_file expdata/opinion1.cfg --thread 1 > opinion1.log 2>&1 &
nohup /data/disk1/zrr/anaconda3/bin/python -u  driver/Train.py --config_file expdata/opinion2.cfg --thread 1 > opinion2.log 2>&1 &
nohup /data/disk1/zrr/anaconda3/bin/python -u  driver/Train.py --config_file expdata/opinion3.cfg --thread 1 > opinion3.log 2>&1 &
nohup /data/disk1/zrr/anaconda3/bin/python -u  driver/Train.py --config_file expdata/opinion4.cfg --thread 1 > opinion4.log 2>&1 &

