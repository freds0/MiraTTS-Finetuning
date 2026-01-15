# Guia de Requirements

Este projeto possui múltiplos arquivos de requirements para diferentes cenários de uso.

## 📦 Arquivos de Requirements Disponíveis

### 1. `requirements.txt` (Principal)
**Uso:** Instalação completa para treinamento local com GPU

```bash
pip install -r requirements.txt
```

**Inclui:**
- PyTorch com suporte CUDA
- Transformers e dependências de treinamento
- Unsloth, TRL, bitsandbytes para otimização
- Bibliotecas de áudio (librosa, soundfile)
- Dependências MiraTTS específicas

**Requisitos:**
- GPU NVIDIA com CUDA 11.8+
- 16GB+ VRAM (recomendado)
- Linux/Windows com CUDA instalado

---

### 2. `requirements-colab.txt` (Google Colab)
**Uso:** Para usar no Google Colab

```python
# No Colab
!pip install -r requirements-colab.txt
```

**Diferenças:**
- Não inclui PyTorch (já pré-instalado no Colab)
- Otimizado para ambiente Colab
- Versões compatíveis com runtime do Colab

**Vantagens:**
- ✅ Ambiente pronto e testado
- ✅ GPU gratuita (T4)
- ✅ Sem problemas de compatibilidade

---

### 3. `requirements-dev.txt` (Desenvolvimento)
**Uso:** Para desenvolvimento local com ferramentas extras

```bash
pip install -r requirements-dev.txt
```

**Inclui tudo de requirements.txt mais:**
- Ferramentas de qualidade de código (black, flake8, pylint)
- Testing (pytest)
- Jupyter notebooks
- Documentação (sphinx)
- Debugging e profiling

**Para quem:**
- Desenvolvedores contribuindo para o projeto
- Análise e debugging de código
- Criação de documentação

---

### 4. `requirements-cpu.txt` (CPU apenas)
**Uso:** Ambiente sem GPU (apenas inferência/testes leves)

```bash
pip install -r requirements-cpu.txt
```

**Diferenças:**
- Sem bitsandbytes, xformers, unsloth
- onnxruntime (CPU version)
- PyTorch sem CUDA

**⚠️ Limitações:**
- Treinamento será EXTREMAMENTE LENTO
- Recomendado APENAS para:
  - Teste de carregamento de dados
  - Inferência com modelo já treinado
  - Desenvolvimento de código (não treino)

---

### 5. `requirements-minimal.txt` (Mínimo)
**Uso:** Apenas para testar carregamento de dados

```bash
pip install -r requirements-minimal.txt
```

**Inclui apenas:**
- PyTorch básico
- Datasets e áudio (librosa, soundfile)
- HuggingFace essenciais

**Para:**
- Verificar se o dataset carrega corretamente
- Testar scripts de processamento de dados
- Desenvolvimento de data loaders

**Exemplo:**
```bash
python test_ljspeech_simple.py
```

---

## 🚀 Guia de Instalação Rápida

### Para Treinamento (Recomendado: Colab)

#### Opção A: Google Colab
```python
!git clone <seu-repo>
%cd <seu-repo>
!pip install -r requirements-colab.txt
```

#### Opção B: Ambiente Local com GPU
```bash
# Criar ambiente
conda create -n miratts python=3.10 -y
conda activate miratts

# Instalar PyTorch com CUDA primeiro
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Instalar resto das dependências
pip install -r requirements.txt
```

### Para Desenvolvimento

```bash
conda create -n miratts-dev python=3.10 -y
conda activate miratts-dev
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements-dev.txt
```

### Para Teste de Dataset (sem treino)

```bash
python -m venv venv_test
source venv_test/bin/activate  # Linux/Mac
# ou
venv_test\Scripts\activate  # Windows

pip install -r requirements-minimal.txt
python test_ljspeech_simple.py
```

---

## 🔍 Verificação da Instalação

Após instalar, verifique se tudo está funcionando:

### 1. Verificar PyTorch e CUDA
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### 2. Verificar Transformers
```bash
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 3. Verificar Unsloth (se aplicável)
```bash
python -c "import unsloth; print('Unsloth OK')"
```

### 4. Verificar áudio
```bash
python -c "import librosa, soundfile; print('Audio libs OK')"
```

### 5. Teste completo
```bash
python test_ljspeech_simple.py
```

Se todos os testes passarem, você está pronto!

---

## 📋 Matriz de Compatibilidade

| Arquivo | PyTorch | CUDA | GPU | Treinamento | Inferência | Desenvolvimento |
|---------|---------|------|-----|-------------|------------|-----------------|
| requirements.txt | ≥2.8.0 | ✅ | ✅ | ✅ | ✅ | ❌ |
| requirements-colab.txt | Auto | ✅ | ✅ | ✅ | ✅ | ❌ |
| requirements-dev.txt | ≥2.8.0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| requirements-cpu.txt | ≥2.8.0 | ❌ | ❌ | 🐢 | ✅ | ❌ |
| requirements-minimal.txt | ≥2.8.0 | ❌ | ❌ | ❌ | ❌ | ❌ |

**Legenda:**
- ✅ Suportado
- ❌ Não suportado
- 🐢 Muito lento (não recomendado)

---

## 🛠️ Troubleshooting

### Erro: "No module named 'X'"
```bash
pip install X
```

### Erro: "CUDA not available"
Verifique:
1. Drivers NVIDIA instalados: `nvidia-smi`
2. PyTorch instalado com CUDA: `python -c "import torch; print(torch.version.cuda)"`
3. Se necessário, reinstale PyTorch com CUDA

### Erro: "RuntimeError: operator torchvision::nms does not exist"
Reinstale torch e torchvision com versões compatíveis:
```bash
pip uninstall torch torchvision torchaudio -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Erro com dependências do git
```bash
pip install git+https://github.com/ysharma3501/FastBiCodec.git --force-reinstall
pip install git+https://github.com/ysharma3501/FlashSR.git --force-reinstall
```

### Memory issues
Use versões quantizadas ou reduza batch size:
```bash
python train_ljspeech.py --batch-size 1 --num-samples 5
```

---

## 📚 Recursos Adicionais

- [INSTALL.md](INSTALL.md) - Guia completo de instalação
- [LJSPEECH_TRAINING.md](LJSPEECH_TRAINING.md) - Guia de treinamento com LJSpeech
- [README.md](README.md) - Documentação geral do projeto

---

## 💡 Recomendações

1. **Para iniciantes:** Use `requirements-colab.txt` no Google Colab
2. **Para treinamento sério:** Use `requirements.txt` em máquina local com GPU forte
3. **Para desenvolvimento:** Use `requirements-dev.txt`
4. **Para testar dados:** Use `requirements-minimal.txt`
5. **Sem GPU:** Não tente treinar, use apenas para inferência com `requirements-cpu.txt`

---

**Última atualização:** 2026-01-15
