# Treinamento com LJSpeech Dataset

## Status Atual

O repositório foi configurado com sucesso para usar o dataset LJSpeech local, mas há um problema de compatibilidade entre as versões de `transformers`, `torch` e `torchvision` que precisa ser resolvido antes do treinamento.

## O que foi implementado

### 1. LJSpeech Loader ([src/mira_tts/ljspeech_loader.py](src/mira_tts/ljspeech_loader.py))

Módulo para carregar o dataset LJSpeech do diretório local:
- Lê o arquivo `metadata.csv`
- Carrega os áudios da pasta `wavs/`
- Cria um HuggingFace Dataset compatível
- Suporta limitação de número de amostras

**Teste bem-sucedido:**
```bash
python test_ljspeech_simple.py
```

Resultado: 10 amostras carregadas com sucesso, incluindo:
- Texto transcrito
- Caminho do áudio
- Taxa de amostragem (16kHz)
- Duração do áudio

### 2. Script de Treinamento ([train_ljspeech.py](train_ljspeech.py))

Script completo para treinamento com LJSpeech:

```bash
python train_ljspeech.py \
    --ljspeech-path /home/fred/Projetos/DATASETS/LJSpeech-1.1/ \
    --num-samples 20 \
    --max-steps 100 \
    --output-dir outputs_ljspeech
```

**Parâmetros disponíveis:**
- `--ljspeech-path`: Caminho para o dataset LJSpeech
- `--num-samples`: Número de amostras para treinar
- `--max-steps`: Passos máximos de treinamento
- `--learning-rate`: Taxa de aprendizado
- `--batch-size`: Tamanho do batch
- `--output-dir`: Diretório de saída
- `--push-to-hub`: Fazer upload para HuggingFace Hub
- `--hub-repo`: Nome do repositório no Hub
- `--hf-token`: Token do HuggingFace

## Problema Atual: Incompatibilidade de Dependências

### Erro:
```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'is_comet_available'
```

### Causa:
Incompatibilidade entre as versões de:
- `transformers==4.56.2`
- `torch>=2.8.0`
- `torchvision`
- `unsloth`

### Soluções Possíveis:

#### Opção 1: Usar ambiente Colab (Recomendado)
O notebook original foi feito para Colab, que tem as dependências corretas pré-instaladas:
1. Fazer upload do código para Google Colab
2. Usar o dataset do Google Drive ou fazer upload
3. Executar o treinamento no ambiente Colab

#### Opção 2: Ajustar versões localmente
```bash
# Desinstalar versões conflitantes
pip uninstall torch torchvision transformers -y

# Reinstalar com versões compatíveis
pip install torch==2.5.0 torchvision==0.20.0
pip install transformers==4.46.0
pip install unsloth
```

#### Opção 3: Usar Docker
Criar um container com as dependências corretas do notebook original.

#### Opção 4: Ambiente virtual limpo
```bash
# Criar novo ambiente
conda create -n miratts python=3.10
conda activate miratts

# Instalar dependências do zero
pip install -r requirements.txt
```

## Estrutura do Dataset LJSpeech

```
/home/fred/Projetos/DATASETS/LJSpeech-1.1/
├── metadata.csv           # Arquivo com transcrições
├── wavs/                  # Diretório com arquivos de áudio
│   ├── LJ001-0001.wav
│   ├── LJ001-0002.wav
│   └── ...
├── train.csv             # Split de treino
└── test.csv              # Split de teste
```

### Formato do metadata.csv:
```
file_id|text_raw|text_normalized
LJ001-0001|Printing, in...|Printing, in...
```

## Próximos Passos

### 1. Resolver dependências
Escolher uma das opções acima para resolver o conflito de versões.

### 2. Testar treinamento com poucas amostras
```bash
python train_ljspeech.py --num-samples 5 --max-steps 10
```

### 3. Treinamento completo
Depois que funcionar, aumentar gradualmente:
```bash
# Pequeno teste
python train_ljspeech.py --num-samples 50 --max-steps 100

# Treinamento médio
python train_ljspeech.py --num-samples 500 --max-steps 1000

# Treinamento completo (13100 amostras)
python train_ljspeech.py --num-samples 13100 --max-steps 5000
```

### 4. Avaliar resultados
```bash
python test_model.py \
    --model-path outputs_ljspeech/ \
    --text "Hello, this is a test." \
    --audio-file /home/fred/Projetos/DATASETS/LJSpeech-1.1/wavs/LJ001-0001.wav \
    --output test_result.wav
```

## Recursos Computacionais Recomendados

Para o dataset LJSpeech completo:
- **GPU:** Mínimo 16GB VRAM (RTX 3090, A100, V100)
- **RAM:** 32GB+
- **Armazenamento:** 10GB+ livre
- **Tempo:** 2-8 horas (dependendo da GPU)

Para testes (50-100 amostras):
- **GPU:** 8GB+ VRAM (RTX 3060, T4)
- **RAM:** 16GB
- **Tempo:** 10-30 minutos

## Notas

1. O LJSpeech tem ~13.100 amostras de áudio
2. Cada amostra tem duração variável (1-10 segundos)
3. O modelo MiraTTS suporta até 30 segundos de áudio (`MAX_SEQ_LENGTH = 30 * 50`)
4. Taxa de amostragem padrão: 16kHz
5. O treinamento usa float32 (não fp16) para evitar NaNs

## Referências

- Dataset LJSpeech: https://keithito.com/LJ-Speech-Dataset/
- Modelo MiraTTS: https://huggingface.co/YatharthS/MiraTTS
- Unsloth: https://github.com/unslothai/unsloth
