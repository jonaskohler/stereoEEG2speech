#!/bin/bash
counter=6
gpu=6

waittime=1

bs=521
tfr=0.1

epochs=110
swas=140

ws=333
lr=0.001

hs=333
dro=0.1

pnpndim=256

for tfr in 0.1 
do
	for dro in 0.1
		do
		for hs in 333
		do
			screen -S "run${counter}" -dm bash -c "sleep ${waittime}; CUDA_VISIBLE_DEVICES=$gpu python3 eeg_main.py \
			--SWA=False --swa_start $swas \
			--dropout $dro --convolve_eeg_1d=False --convolve_eeg_2d=True --convolve_eeg_3d=False \
			--use_bahdanau_attention=True \
			--OLS=False --DenseModel=False \
			--use_MFCCs=True --discretize_MFCCs=False \
			--hidden_size $hs --batch_size $bs \
			--pre_and_postnet=True --pre_and_postnet_dim $pnpndim\
			--epochs $epochs \
			--learning_rate $lr --window_size $ws  \
			--teacher_forcing_ratio $tfr --debug=False \
			--mixed_loss=False \
			--patient_eight=False \
			--patient_thirteen=True \
			--double_trouble=False; \
			exec sh"
		    counter=$((counter + 1))
		    waittime=$((counter + 25)) # s.t. not all runs grab the same GPU
		    gpu=$((gpu + 1))
		done
	done
done
