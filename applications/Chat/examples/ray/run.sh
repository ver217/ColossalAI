for pretrain in "facebook/opt-350m" "facebook/opt-1.3b" "facebook/opt-2.7b" "facebook/opt-6.7b" "facebook/opt-13b"
do
    for experience_steps in 8 16
    do
        for experience_batch_size in 4 8 16
        do
            for update_steps in 4 16
            do
                for train_batch_size in 4 8 16
                do
                nohup python mmmt_prompt.py \
                    --prompt_path /home/lccsr/data3/awesome-chatgpt-prompts/prompts.csv \
                    --trainer_strategy colossalai_gemini --maker_strategy naive \
                    --model 'opt' \
                    --critic_pretrain "facebook/opt-125m" \
                    --pretrain $pretrain \
                    --num_trainers 4 \
                    --num_makers 4 \
                    --experience_steps $experience_steps \
                    --experience_batch_size $experience_batch_size \
                    --update_steps $update_steps \
                    --train_batch_size $train_batch_size \
                    --debug > logs/output_4_4_pretrain_${pretrain##*/}_experience_steps_${experience_steps}_experience_batch_size_${experience_batch_size}_update_steps_${update_steps}_train_batch_size_${train_batch_size}.txt 2>&1
                done
            done
        done
    done
done