#!/bin/bash
# List of MINDS classes
classes=('Aluno' 'Bala' 'Espelho' 'Cinco' 'Sapo' 'Aproveitar' 'Conhecer' 'Acontecer' 'Maca' 'Vontade' 'Vacina' 'Banco' 'Esquina' 'Barulho' 'Filho' 'Amarelo' 'Banheiro' 'Medo' 'America' 'Ruim')

# Directory containing the data
data_dir="../MINDS/"

# Iterate over each class and run the Python script
for class in "${classes[@]}"
do
    echo "Processing class: $class"
    #nohup python build_tensor_dataset.py --data_dir "$data_dir" -n "$class" > logs/"$class".log &
    # nohup python build_tensor_dataset.py --data_dir "$data_dir" -n "$class" --frames 32 > logs/"$class".log &
    python build_tensor_dataset.py --data_dir "$data_dir" --frames 32 -n "$class" 
done
