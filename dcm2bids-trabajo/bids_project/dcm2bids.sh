directorio_principal="sourcedata/"

for carpeta in "$directorio_principal"/*; do
    prefijo_sujeto=$(basename "$carpeta")
    dcm2bids -d "$carpeta" -p "$prefijo_sujeto" -c code/dcm2bids_config.json --auto_extract_entities -o Tumor/
done