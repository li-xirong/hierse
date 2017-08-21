source common.ini

for embedding in conse conse2 hierse hierse2
do    
    label_vec_name=${w2v_corpus},${w2v},${embedding}
    python ../build_label_vec.py ${label_set} --w2v_corpus ${w2v_corpus} --w2v ${w2v} --embedding $embedding --overwrite $overwrite
    python ../im2vec.py $test_collection $pY0 --label_vec_name $label_vec_name --overwrite $overwrite
done
