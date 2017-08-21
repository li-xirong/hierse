source common.ini

label_sets=( $Y0 $Y1 )

for embedding in conse conse2 hierse hierse2
do 
    label_vec_name=${w2v_corpus},${w2v},${embedding}
    
    for label_set in "${label_sets[@]}"
    do
        python ../build_label_vec.py ${label_set} --w2v_corpus ${w2v_corpus} --w2v ${w2v} --embedding $embedding --overwrite $overwrite
    done

    python ../im2vec.py $test_collection $pY0 --label_vec_name $label_vec_name --overwrite $overwrite
    python ../zero_shot_tagging.py $test_collection --Y0 $Y0 --Y1 $Y1 --pY0 $pY0 --w2v_corpus ${w2v_corpus} --w2v ${w2v} --embedding $embedding --overwrite $overwrite
done
