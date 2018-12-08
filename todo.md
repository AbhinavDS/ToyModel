# TODO: 
## On Toy Model 1
 Batching almost done
 General number of vertex
 Blocks usage

## On Toy Model 2
 Move to multiple polygons case
 Decide on Input Done
 Decide on genus
 Do Splitting
 Do Merging instead of GENUS supervision
 Do RNN
 Think if other losses are needed
## Reward based sampling from buffer
## Check on eagle data

## On Toy Model 3
 Sanity check if extension works on 2d torus 


 #### 

 DDPG - TONIGHT DONE
 PLANE - GEN - DONE
 TRAIN ON 100 - STARTED
 TRAIN ON 10000 or 1000 - STARTED FOR 1000
 FIX ADDER - NO NEED SINCE WE WILL DETERMINISTICALLY ADD IT
 HOW TO TRAIN and gen data for RL ON MULTIPLE POLYGON
 HOW TO TERMINATE SPLIT

#### 
SAVE MODEL, TEST CODE
SAMPLE
PUT CORRECT MODEL IN RL
###
WRITE SPLIT AND FURTHER DEFORM

# Current command
python3 main.py  --batch_size 100 -n 2000 -d 3 --data_dir data/2_1000 -s ckpt/model_2_1000.toy -sf 10000

python3 main.py  --batch_size 1 -n 2000 -d 3 --data_dir data/1_plane -s ckpt/model_1_plane.toy -sf 1_plane

python3 main.py  --batch_size 1 -n 2000 -d 3 --data_dir data/2 -s ckpt/model_2_1000.toy -sf 1



####
python3 main.py  --batch_size 1 -n 2000 --data_dir data/1_plane -s ckpt/model_1_plane.toy -sf 1_plane  -d 3 -lr 1e-6 --lambda_lap 2 --lambda_n 1

####
TODO
Plane 10000 data
python3 main.py  -n 2000 --data_dir data/10000_plane -s ckpt/model_10000_plane.toy -sf 10000_plane  -d 5 -lr 1e-6 --lambda_lap 2 --lambda_n 1

Plane 2000
python3 main.py  --batch_size 100 -n 2000 -d 3 --data_dir data/2_1000 -s ckpt/model_2_1000.toy -sf 10000





python3 main.py  --batch_size 10 -n 2000 -d 3 --data_dir data/2 -s ckpt/model_2_1000.toy -l ckpt/model_2_1000.toy -sf 1000 -lr 0.00000512
