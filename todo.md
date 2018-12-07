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

 DDPG - TONIGHT
 EAGLE
 TRAIN ON 100 - STARTED
 TRAIN ON 10000 or 1000 - STARTED FOR 1000
 FIX ADDER - NO NEED SINCE WE WILL DETERMINISTICALLY ADD IT
 HOW TO TRAIN and gen data for RL ON MULTIPLE POLYGON
 HOW TO TERMINATE SPLIT


# Current command
python3 main.py  --batch_size 100 -n 2000 -d 3 --data_dir data/2_1000 -s ckpt/model_2_1000.toy -sf 10000
