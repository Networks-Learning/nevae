#!/bin/bash

touch /tmp/testRMSEcharacter.txt
rm /tmp/testRMSEcharacter.txt

simulations=1
for i in `seq $simulations`
do
    echo $i
    cat ../../../data/drug_like/lowPH/training/BO_30/nohup.BO.out | grep RMSE | head -1 | cut -d " " -f 4 >> /tmp/testRMSEcharacter.txt
done

mean_RMSE_character=`python -c "import numpy as np; print(np.mean(np.loadtxt('/tmp/testRMSEcharacter.txt')))"`

std_RMSE_character=`python -c "import numpy as np; print(np.std(np.loadtxt('/tmp/testRMSEcharacter.txt')) / np.sqrt(10))"`

echo RMSE character: $mean_RMSE_character $std_RMSE_character

touch /tmp/testRMSEgrammar.txt
rm /tmp/testRMSEgrammar.txt
touch /tmp/testRMSEcharacter.txt
rm /tmp/testRMSEcharacter.txt
for i in `seq $simulations`
do
    echo $i
    cat ../../../data/drug_like/lowPH/training/BO_30/nohup.BO.out |grep "Test ll" | grep -v erro | head -1 | cut -d " " -f 4 >> /tmp/testRMSEcharacter.txt
done

mean_RMSE_character=`python -c "import numpy as np; print(np.mean(np.loadtxt('/tmp/testRMSEcharacter.txt')))"`

std_RMSE_character=`python -c "import numpy as np; print(np.std(np.loadtxt('/tmp/testRMSEcharacter.txt')) / np.sqrt(10))"`

echo LL character: $mean_RMSE_character $std_RMSE_character

rm /tmp/testRMSEcharacter.txt
