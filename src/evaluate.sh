#!/bin/bash
max_seq='3'
recompute='1'
train_steps='4'
min_ratio_i='1'
max_ratio_i='1'
archs=(FC Probabilistic Transformer LSTM)
gen=(FC_sent Transformer_sent LSTM_sent)
for i in `seq $min_ratio_i $max_ratio_i`;
	do real_ratio=$(printf "%.1f" $(bc -l <<< "-$i/10+1"))
	echo "[INFO]Computing results for real_ratio $real_ratio"
	# split for cross validation
	for split in `seq 0 $max_seq`;
		do echo "[INFO]Split n°$split"
		num_arch='0'
		# doc classifier loop
		for doc_classifier in "${archs[@]}" ; do
			echo "[INFO]$doc_classifier classifier"
			num_arch=$((num_arch+1))
			# generator's arch loop
			for gen_arch in "${gen[@]}" ; do

				if [ $gen_arch != LSTM_sent ]; then
					next_real_ratio=$real_ratio
					next_split=$split
					next_doc_classifier=$doc_classifier
					next_gen_arch=${gen[$num_arch]}
				elif [ $doc_classifier != LSTM ]; then
					next_real_ratio=$real_ratio
					next_split=$split
					next_doc_classifier=${archs[$num_arch]}
					next_gen_arch=${gen[0]}
				elif [ $split != $max_seq ]; then
					next_real_ratio=$real_ratio
					next_split=$(bc -l <<< "$split+1")
					next_doc_classifier=${archs[0]}
					next_gen_arch=${gen[0]}
				else
					next_real_ratio=$(printf "%.1f" $(bc -l <<< "-$(bc -l <<< "$i+1")/10+1"))
					next_split='0'
					next_doc_classifier=${archs[0]}
					next_gen_arch=${gen[0]}
				fi
				FILE=../results/results_${next_gen_arch}_${next_doc_classifier}_${next_real_ratio}_$next_split.txt
				echo "[INFO]Current file: ../results/results_${gen_arch}_${doc_classifier}_${real_ratio}_$split.txt"
				echo "[INFO]Next file: $FILE"
				
				if [ ! -f "$FILE" ]; then
			    	python3 sort.py -real_ratio $real_ratio -num_split $split
					python3 train.py -mode train -parts_of_interest 'STATE_OF_THE_ART' \
					-max_pos 1500 -need_preprocessing True -dataset 'train' 'valid' \
					-real_ratio $real_ratio -num_split $split -only_preprocessing True
					python3 train.py -mode train -parts_of_interest 'STATE_OF_THE_ART' \
					-lr 1e-5 -visible_gpus 0,1,2,3 -train_steps $train_steps -max_pos 1500 -finetune_bert True \
					-need_preprocessing False -classification 'separate' \
					-g_learning_rate 1e-5 -batch_size 4 -test_batch_size 40 -generator $gen_arch \
					-doc_classifier $doc_classifier -dataset 'train' 'valid' -real_ratio $real_ratio \
					-num_split $split -evaluate_x_steps 20
				else
					echo "[INFO]Skipping..."
				fi
			done
		done
	done
done

# Display results
python3 gather_results.py
