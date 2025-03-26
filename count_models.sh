for (( i=0; i<=15; i+=3 ))
do
    echo icl=$i
    ls ./models/Mistral-7B-Instruct-v0.3_ft_* | grep icl=$i | wc -l
done