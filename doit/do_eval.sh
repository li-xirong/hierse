source common.ini

for embedding in conse conse2 hierse hierse2
do 
    method=$pY0/${w2v_corpus},${w2v},${embedding}
    python ../evaluate.py $test_collection $method
done
