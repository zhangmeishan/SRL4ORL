nohup /data/disk1/zrr/anaconda3/bin/python -u  driver/Train.py --config_file expdata/opinion1.cfg --srl_config_file expdata/srl.cfg --thread 1 > opinion1.joint.log 2>&1 &
nohup /data/disk1/zrr/anaconda3/bin/python -u  driver/Train.py --config_file expdata/opinion2.cfg --srl_config_file expdata/srl.cfg --thread 1 > opinion2.joint.log 2>&1 &
nohup /data/disk1/zrr/anaconda3/bin/python -u  driver/Train.py --config_file expdata/opinion3.cfg --srl_config_file expdata/srl.cfg --thread 1 > opinion3.joint.log 2>&1 &
nohup /data/disk1/zrr/anaconda3/bin/python -u  driver/Train.py --config_file expdata/opinion4.cfg --srl_config_file expdata/srl.cfg --thread 1 > opinion4.joint.log 2>&1 &

