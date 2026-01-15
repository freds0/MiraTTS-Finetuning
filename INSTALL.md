# Guia de Instalação

## Problema de Compatibilidade Identificado

Durante os testes, identificamos um conflito entre as versões de `transformers`, `torch` e `torchvision`. Este guia fornece soluções.

## Solução Recomendada: Ambiente Colab

O código original foi desenvolvido para Google Colab. Esta é a maneira mais fácil de começar:

### 1. Acesse o Google Colab
https://colab.research.google.com/

### 2. Faça upload do código
```python
# No Colab, clone ou faça upload dos arquivos
!git clone <seu-repositorio>
# ou use o menu Files para fazer upload
```

### 3. Configure o dataset
```python
# Fazer upload do LJSpeech para Google Drive ou usar wget
!wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
!tar -xjf LJSpeech-1.1.tar.bz2
```

### 4. Execute o treinamento
```python
!python train_ljspeech.py --ljspeech-path ./LJSpeech-1.1 --num-samples 20
```

## Solução Alternativa 1: Ambiente Virtual Limpo

### Com Conda (Recomendado)

```bash
# Criar novo ambiente
conda create -n miratts python=3.10 -y
conda activate miratts

# Instalar PyTorch com CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Instalar dependências principais
pip install transformers==4.46.0
pip install accelerate==1.8.1
pip install datasets>=3.4.1,<4.0.0
pip install sentencepiece protobuf
pip install huggingface_hub>=0.34.0
pip install hf_transfer

# Instalar Unsloth e TRL
pip install unsloth trl peft bitsandbytes

# Instalar dependências de áudio
pip install librosa soundfile
pip install omegaconf einx torchcodec

# Instalar dependências específicas do MiraTTS
pip install onnxruntime-gpu
pip install git+https://github.com/ysharma3501/FastBiCodec.git
pip install git+https://github.com/ysharma3501/FlashSR.git

# Navegar para o diretório do projeto
cd /home/fred/Projetos/Mira-TTS-Finetuning

# Testar carregamento do dataset
python test_ljspeech_simple.py

# Treinar
python train_ljspeech.py --num-samples 5 --max-steps 10
```

### Com venv

```bash
# Criar ambiente
python3.10 -m venv venv_miratts
source venv_miratts/bin/activate

# Seguir os mesmos passos de instalação acima (exceto conda)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# ... resto das dependências
```

## Solução Alternativa 2: Docker

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Instalar Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

# Instalar git dependencies
RUN pip install git+https://github.com/ysharma3501/FastBiCodec.git
RUN pip install git+https://github.com/ysharma3501/FlashSR.git

CMD ["/bin/bash"]
```

### Uso:

```bash
# Build
docker build -t miratts .

# Run
docker run --gpus all -it -v /home/fred/Projetos:/workspace miratts

# Dentro do container
cd /workspace/Mira-TTS-Finetuning
python train_ljspeech.py --num-samples 5
```

## Verificação da Instalação

### 1. Verificar CUDA
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### 2. Verificar transformers
```bash
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

### 3. Verificar unsloth
```bash
python -c "import unsloth; print('Unsloth imported successfully')"
```

### 4. Testar carregamento de dados
```bash
python test_ljspeech_simple.py
```

Se todos os comandos acima funcionarem, você está pronto para treinar!

## Troubleshooting

### Erro: "RuntimeError: operator torchvision::nms does not exist"
- **Causa:** Incompatibilidade entre torch e torchvision
- **Solução:** Reinstalar com versões compatíveis (ver acima)

### Erro: "No module named 'librosa'"
```bash
pip install librosa soundfile
```

### Erro: "CUDA out of memory"
- Reduzir `--batch-size` para 1
- Reduzir `--num-samples`
- Usar GPU com mais memória

### Erro: "No module named 'unsloth'"
```bash
pip install unsloth
```

### Erro com FastBiCodec ou FlashSR
```bash
pip install git+https://github.com/ysharma3501/FastBiCodec.git --force-reinstall
pip install git+https://github.com/ysharma3501/FlashSR.git --force-reinstall
```

## Requisitos Mínimos

- **Python:** 3.9-3.11
- **CUDA:** 11.8+ (12.1 recomendado)
- **GPU:** 8GB+ VRAM para testes, 16GB+ para treinamento completo
- **RAM:** 16GB mínimo, 32GB recomendado
- **Espaço:** 20GB+ livre

## Próximos Passos

Depois de instalar com sucesso:

1. Teste o carregamento do dataset:
   ```bash
   python test_ljspeech_simple.py
   ```

2. Execute um treinamento de teste:
   ```bash
   python train_ljspeech.py --num-samples 5 --max-steps 10
   ```

3. Se funcionar, aumente gradualmente os parâmetros

Consulte [LJSPEECH_TRAINING.md](LJSPEECH_TRAINING.md) para detalhes sobre o treinamento.
