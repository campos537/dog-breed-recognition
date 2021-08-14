# Dog Breed Recognition

## Introducao

A ideia desse projeto eh mostrar uma solucao escalavel para o reconhecimento de racas de cachorro. Sendo demonstrado tambem um metodo escalavel de se adicionar novas racas sem ter um tempo custoso de re-treinamento do modelo.

## Requerimentos

Esse projeto foi desenvolvido utilizando:

- Ubuntu 18.04
- Python3.7
- Nvidia GTX 1080 8GB
- Cuda 11.4

para instalar as dependencias eh recomendado seguir os passos abaixo:

```
sudo apt-get install python3-tk
python3.7 -m virtualenv dogs-env
source dogs-env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name dogs-env --display-name "dogs-env"
```

apos isso para replicar o treinamento e o teste dos modelos pode abrir o notebook 
com o comando `jupyter notebook`.

## Etapas

O projeto eh dividido em partes, nessa secao eh explicada cada parte e como rodar a mesma

### Primeira Parte

Na primeira parte eh realizado o treinamento do modelo, para isso primeiramente o dataset teve algumas imagens pretas removidas
restando apenas as imagens dos cachorros, apos isso foram separados os sets de treino e teste (10% do total), dessa forma se garante que as imagens
de teste nao estao sendo usadas no treino.

Para rodar novamente o treinamento entre no notebook `scripts/training-scripts/dogs_breed_research.ipynb` e rode o mesmo com o dataset de treino e teste
localizado na pasta `scripts/training-scripts/`

Para testar os modelos gerados utilize do notebook `dogs_breed_inference.ipynb`. O notebook ira testar cada epoch de cada modelo organizado dessa forma:
```
models/
    resnet50-dogbreed/
        epoch1.pth
        epoch2.pth
        ...
    resnet34-dogbreed/
        epoch1.pth
        epoch2.pth
        ...
```

Ao achar o melhor modelo ele ira gerar o .ONNX do mesmo para utilizacao posterior no OpenCV. Rodando o ultimo Step do notebook eh possivel escolher uma imagem e fazer a inferencia da mesma.

obs: Para gerar as metricas foi utilizado o PyCM que ao final ira mostrar o resultado do melhor modelo alem de gerar um report em html mais completo (por ter muitas classes fica mais confuso de interpretar) com o nome do mesmo.

### Segunda e Terceira Parte