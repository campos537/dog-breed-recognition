# Dog Breed Recognition

## Introducao

A ideia desse projeto eh apresentar modelos e um sistema de reconhecimento de racas de cachorros. Alem disso o projeto implementa a possibilidade de adicionar novas racas de cachorro nao vistas em tempo de treinamento.
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

apos isso tanto os scripts de treinamentos e o dog-breed-system vao ser possiveis de se rodar.

## Etapas

O projeto eh dividido em partes, nessa secao eh explicada cada parte e como rodar a mesma

### Treinamento e Validacao (Classificacao)

Foi realizado o treinamento de dois modelos Resnet50 e Resnet101, para isso primeiramente o dataset teve algumas imagens pretas removidas
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

### Modelo Descriptor

O modelo mostrado anteriormente foi treinado para classificacao possuindo no final a layer Softmax, para criar o modelo pedido na Etapa 2 se substituiu o ultimo bloco do modelo (fc) pela layer Linear dando o output de 512 descriptors, sendo o notebook `scripts/training-scripts/dogs_breed_descriptor.ipynb` utilizado para isso.

### Codigo inferencia

O codigo de inferencia esta localizado na pasta `dog-breed-system/predictor` ele foi implementado utilizando o OpenCV DNN suportando todos os backends e targets que podem ser utilizados pelo mesmo dependendo do seu hardware assim como os frameworks disponiveis pelo Opencv. A classe predictor recebe um dict com o path do modelo e
toda a configuracao de pre-processamento necessario para funcionar, um exemplo de JSON que pode ser utilizado como configuracao pode ser visto abaixo:

```
{
    "framework": "ONNX",
    "trained_file": "/media/disk1/Repositorios/UnicoID/dog-breed-recognition/dog-breed-system/models/dog_class_model_resnet50.onnx",
    "backend": 3,
    "target": 0,
    "input_size": [1,224,224,3],
    "scalefactor": 0.003921569,
    "swapRB": true,
    "labels": ["Afghan_hound", "Airedale", "American_Staffordshire_terrier", "Appenzeller", "Bedlington_terrier", "Blenheim_spaniel", "Border_collie", 
        "Border_terrier", "Boston_bull", "Bouvier_des_Flandres", "Brabancon_griffon", "Brittany_spaniel", "Chesapeake_Bay_retriever", "Chihuahua", 
        "Dandie_Dinmont", "Doberman", "English_foxhound", "English_setter", "EntleBucher", "French_bulldog", "German_shepherd", "German_short", 
        "Gordon_setter", "Great_Dane", "Greater_Swiss_Mountain_dog", "Ibizan_hound", "Irish_setter", "Irish_terrier", "Irish_water_spaniel", 
        "Irish_wolfhound", "Italian_greyhound", "Japanese_spaniel", "Labrador_retriever", "Lakeland_terrier", "Leonberg", "Lhasa", "Maltese_dog", 
        "Newfoundland", "Norfolk_terrier", "Norwegian_elkhound", "Old_English_sheepdog", "Pekinese", "Pembroke", "Pomeranian", "Saint_Bernard", 
        "Saluki", "Scotch_terrier", "Sealyham_terrier", "Shetland_sheepdog", "Shih", "Siberian_husky", "Staffordshire_bullterrier", "Sussex_spaniel", 
        "Tibetan_mastiff", "Tibetan_terrier", "Walker_hound", "Weimaraner", "Welsh_springer_spaniel", "Yorkshire_terrier", "affenpinscher", "basenji", 
        "basset", "beagle", "black", "bloodhound", "bluetick", "borzoi", "boxer", "briard", "bull_mastiff", "cairn", "chow", "clumber", "cocker_spaniel", 
        "collie", "dhole", "dingo", "flat", "giant_schnauzer", "keeshond", "kelpie", "komondor", "kuvasz", "malamute", "malinois", "miniature_pinscher", 
        "miniature_poodle", "otterhound", "papillon", "pug", "schipperke", "silky_terrier", "soft", "standard_poodle", "standard_schnauzer", "toy_poodle", 
        "toy_terrier", "vizsla", "whippet", "wire"]

}
```
### Dog Breed System

#### Python-tk interface

O sistema foi criado para realizar o teste visual das etapas solicitadas, no primeiro campo de texto [Teste Imagem (Parte 1 Classificacao)] voce pode colocar a imagem de input e clicar no botao com pata de cachorro para realizar a inferencia, no segundo campo [Adicione Cachorros (Parte 2 Enroll)] voce pode passar uma pasta
que contenha imagens de cachorros de uma raca especifica e adicionar o nome de raca clicando em adicionar, no terceiro e ultimo campo [Teste Imagem Enroll (Parte 2 Enroll)] voce pode fazer o upload de uma imagem para testar exclusivamente a parte de enroll do projeto.

Ambas a primeira e a terceira parte do sistema reconhece as racas unknowns, baseadas em confianca e similiaridades respectivamente.

Para rodar utilizando o Ubuntu tenha instalado todas as dependencias localizadas no `requirements.txt` e adicione dois arquivos de configuracao dos modelos nos nomes
`config_class.json` e `config_desc.json` seguindo o padrao acima, sendo que para o `config_class.json` o modelo tem que ser de classificacao possuindo as labels no mesmo e para `config_desc.json` o modelo precisa possuir como output um vetor de features sem ter labels adicionadas no mesmo. Apos isso rode utilizando os comandos abaixo:

```
cd dog-breed-system
python dog-breed.py
```

#### Web App interface

O sistema foi criado como exemplo de uma aplicacao web utilizando o sistema de classificacao de cachorros
para rodar a aplicacao siga as mesmas instrucoes de configuracao acima porem rodando os comandos abaixo:

```
cd dog-breed-system
python dog-breed-web.py
```

apos isso abra o seu navegador e digite `0.0.0.0:8080`