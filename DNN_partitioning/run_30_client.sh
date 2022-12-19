rank=('01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12' '13' '14' '15' '01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12' '13' '14' '15' )
ip=('11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' '23' '24' '25' '41' '42' '43' '44' '45' '46' '47' '48' '49' '50' '51' '52' '53' '54' '55')
node_num=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30) 
gnome-terminal -x bash -c "./run_client.sh edge01@192.168.0.11 edge01 'cd /data/zywang/CoopFL_nin_cifar_10_30device;python3.6 device.py --device_num 30 --node_num 1 --use_gpu 1 --model_type 'AlexNet' --dataset_type 'image' ';exec bash;"
sleep 1s
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do
sleep 1s
gnome-terminal -x bash -c "./run_client.sh edge${rank[$i]}@192.168.0.${ip[$i]} edge${rank[$i]} 'cd /data/zywang/CoopFL_nin_cifar_10_30device;python3 device.py --device_num 30 --node_num ${node_num[$i]} --use_gpu 0 --model_type 'AlexNet' --dataset_type 'image' ';exec bash;"
done
for i in 15 16 17 18 19 20 21 22 23 24 25 26
do
sleep 1s
gnome-terminal -x bash -c "./run_client.sh nx${rank[$i]}@192.168.0.${ip[$i]} nx${rank[$i]} 'cd /data/zywang/CoopFL_nin_cifar_10_30device;python3 device.py --device_num 30 --node_num ${node_num[$i]} --use_gpu 0 --model_type 'AlexNet' --dataset_type 'image' ';exec bash;"
done
sleep 1s
gnome-terminal -x bash -c "./run_client.sh nx13@192.168.0.53 nx13 'cd /data/zywang/CoopFL_nin_cifar_10_30device;python3 device.py --device_num 30 --node_num 28 --use_gpu 1 --model_type 'NIN' --dataset_type 'image' ';exec bash;"
sleep 1s
for i in 28 29
do
sleep 1s
gnome-terminal -x bash -c "./run_client.sh nx${rank[$i]}@192.168.0.${ip[$i]} nx${rank[$i]} 'cd /data/zywang/CoopFL_nin_cifar_10_30device;python3 device.py --device_num 30 --node_num ${node_num[$i]} --use_gpu 0 --model_type 'AlexNet' --dataset_type 'image' ';exec bash;"
done
# lsof -i:Pid
# kill -9 

#python PS.py --device_num 30 --edge_number 3 --model_type 'NIN' --dataset_type 'cifar10'
#python edge.py --device_num 30 --model_type 'NIN' --dataset_type 'cifar10'
#python edge1.py --device_num 30 --model_type 'NIN' --dataset_type 'cifar10'
