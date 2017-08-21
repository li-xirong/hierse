source common.ini

label_sets=( $Y0 $Y1 )

for label_set in "${label_sets[@]}"
do
    for embedding in conse conse2 hierse hierse2
    do
        python ../build_label_vec.py ${label_set} --w2v_corpus ${w2v_corpus} --w2v ${w2v} --embedding $embedding --overwrite $overwrite
    done
done




