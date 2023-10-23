#model="LSTMClassifier_32_3"
evaluate="False"
#model="LSTMClassifier_96_3"
#model="LSTMClassifier_96_2"
#varName="trigger"
#model="LSTM_32_3"
model="LSTMClassifier_Reg_32_3"
#model="LSTMClassifier_Reg_16_1"
#model="LSTMClassifier_Reg_64_3"
#model="LSTM_32_3"
#model="LSTM_96_3"
#varName="rthcuten"
#varName="nca"
varName="all"
norm_method="abs-max"
#norm_method="z-score"
gpu="0,1,2,3"
num_gpu=4
#
#default value
#learning_rate=1e-3
learning_rate=3e-3

weights='1'
weights='specified'

#error_train=mse_trigger
error_train=mae_trigger

sub_folder=${varName}_${norm_method}_${error_train}_${weights}
#sub_folder=${varName}

load_model="False"
load_checkpoint_name="temp"
resume_epoch=-1
#dropout=0.2
dropout=0
#resume_epoch=130
#load_model="True"
#evaluate="True"
load_checkpoint_name=$(ls -lrt checkpoints/${model}/${sub_folder}/ | tail -n 1|awk '{print $9}').tar

main_folder=${model}
if [ "${dropout}" != "0" ];then
	main_folder=${model}_drop${dropout}
fi

#sudo /home/data2/RUN/DAMO/xiaohui/anaconda3/bin/python ../main.py \
#sudo nohup /home/data2/RUN/DAMO/xiaohui/anaconda3/bin/python ../main.py >./logs/wrf_${model}_${varName}_${norm_method}_${error_train}_${weights}.log 2>&1 \
sudo nohup /home/data2/RUN/DAMO/xiaohui/anaconda3/bin/python ../main.py >./logs/wrf_${model}_${varName}_${norm_method}_${error_train}_${weights}_epoch200.log 2>&1 \
--main_folder ${main_folder} \
--sub_folder ${sub_folder} \
--prefix ${model}_${varName} \
--dataset_type "WRF" \
--loss_type "v01" \
--learning_rate  ${learning_rate}  \
--batch_size 1 \
--gpu ${gpu} \
--num_gpu ${num_gpu} \
--model_name ${model} \
--dropout ${dropout} \
--num_workers 10 \
--num_epochs 201 \
--save_model "True" \
--save_checkpoint_name "model" \
--save_per_samples 10000 \
--load_model ${load_model} \
--load_checkpoint_name ${load_checkpoint_name} \
--resume_epoch ${resume_epoch} \
--evaluate ${evaluate} 

