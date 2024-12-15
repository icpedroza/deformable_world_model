#!/usr/bin/env bash

python control.py \
	--stage control \
	--data_type ngrip_fixed \
	--model_path dump/dump_ngrip_fixed/out_12-Dec-2024-18:02:59.297609/net_epoch_19_iter_34.pth \
	--shooting_size 200 \
	--control_algo predict \
	--n_grips 5 \
	--predict_horizon 2 \
	--opt_algo GD \
	--correction 1 \
	--shape_type alphabet \
	--goal_shape_name A \
	--debug 1
