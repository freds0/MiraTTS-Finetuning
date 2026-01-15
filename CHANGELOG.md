# Changelog

Todas as mudanças importantes neste projeto serão documentadas neste arquivo.

## [0.1.0] - 2026-01-15

### Adicionado

#### Estrutura Modular
- Transformação completa do notebook Jupyter em repositório modularizado
- Criado pacote Python `mira_tts` com módulos separados por função
- Estrutura de diretórios organizada (`src/`, `scripts/`, `utils/`)

#### Módulos Core
- `config.py` - Gerenciamento centralizado de configurações
- `model_loader.py` - Carregamento e inicialização do modelo MiraTTS
- `audio_codec.py` - Codificação e decodificação de áudio com TTSCodec
- `data_processor.py` - Processamento de datasets HuggingFace
- `trainer.py` - Lógica de treinamento com SFTTrainer
- `inference.py` - Geração de áudio e inferência
- `ljspeech_loader.py` - Carregador específico para dataset LJSpeech local

#### Scripts Executáveis
- `train.py` - Script principal para treinamento com datasets HuggingFace
- `train_ljspeech.py` - Script para treinamento com LJSpeech local
- `test_model.py` - Script de inferência e testes
- `test_ljspeech_simple.py` - Teste de carregamento do dataset LJSpeech
- `test_ljspeech_loader.py` - Teste completo do loader LJSpeech

#### Documentação
- `README.md` - Documentação principal do repositório
- `REQUIREMENTS.md` - Guia detalhado de requirements e instalação
- `INSTALL.md` - Guia de instalação e troubleshooting
- `LJSPEECH_TRAINING.md` - Guia específico para treinamento com LJSpeech
- `CHANGELOG.md` - Este arquivo

#### Requirements
- `requirements.txt` - Dependências completas para GPU (produção)
- `requirements-colab.txt` - Dependências otimizadas para Google Colab
- `requirements-cpu.txt` - Dependências para ambientes CPU-only
- `requirements-dev.txt` - Dependências de desenvolvimento
- `requirements-minimal.txt` - Dependências mínimas para testes

#### Configuração
- `setup.py` - Configuração do pacote Python
- `.gitignore` - Arquivos a serem ignorados pelo Git

### Funcionalidades

#### Suporte a Datasets
- ✅ Datasets HuggingFace (via `datasets` library)
- ✅ Dataset LJSpeech local (carregamento direto de arquivos)
- ✅ Processamento automático de áudio (16kHz)
- ✅ Tokenização e encoding para treinamento

#### Treinamento
- ✅ Configurações flexíveis via CLI ou arquivo de config
- ✅ Suporte a full finetuning com float32
- ✅ Monitoramento de uso de GPU/VRAM
- ✅ Checkpoint automático
- ✅ Upload para HuggingFace Hub

#### Inferência
- ✅ Geração de áudio a partir de texto e referência
- ✅ Parâmetros ajustáveis (temperature, top_k, top_p)
- ✅ Salvamento de áudio em formato WAV (48kHz)

#### Interface
- ✅ CLI completa com argumentos
- ✅ API Python para uso programático
- ✅ Exemplos de uso documentados

### Testado

#### Testes Realizados
- ✅ Carregamento do dataset LJSpeech local
  - 10 amostras carregadas com sucesso
  - Áudio em 16kHz
  - Textos transcritos corretamente
  - Durações variáveis (1.9s - 9.6s)

#### Ambiente de Teste
- Sistema: Linux (Ubuntu)
- Python: 3.12
- Dataset: LJSpeech-1.1 (local)
- Localização: `/home/fred/Projetos/DATASETS/LJSpeech-1.1/`

### Problemas Conhecidos

#### Dependências
- ⚠️ Conflito entre `transformers 4.56.2`, `torch 2.8+` e `torchvision`
  - Erro: `RuntimeError: operator torchvision::nms does not exist`
  - **Solução:** Usar Google Colab ou criar ambiente virtual limpo
  - Documentado em `INSTALL.md`

#### Requisitos
- Requer GPU com 16GB+ VRAM para treinamento completo
- Treinamento em CPU não é viável (extremamente lento)
- Modelo funciona apenas em float32 ou bfloat16 (NaNs em fp16)

### Melhorias Futuras

#### Planejado
- [ ] Resolver conflito de dependências para ambiente local
- [ ] Adicionar suporte a outros datasets (LibriTTS, VCTK, etc.)
- [ ] Implementar avaliação automática de qualidade
- [ ] Adicionar métricas de treinamento (MOS, MCD, etc.)
- [ ] Criar interface web (Gradio/Streamlit)
- [ ] Suporte a treinamento distribuído
- [ ] Quantização e otimização de modelos
- [ ] Testes unitários automatizados

#### Considerado
- [ ] Suporte a LoRA/QLoRA para reduzir uso de memória
- [ ] Fine-tuning seletivo de camadas
- [ ] Data augmentation para áudio
- [ ] Multi-speaker training
- [ ] Voice cloning com poucas amostras

## Estrutura de Versão

Este projeto segue [Semantic Versioning](https://semver.org/):
- **MAJOR**: Mudanças incompatíveis na API
- **MINOR**: Novas funcionalidades (compatíveis)
- **PATCH**: Correções de bugs

## Categorias de Mudanças

- **Adicionado**: Novas funcionalidades
- **Modificado**: Mudanças em funcionalidades existentes
- **Depreciado**: Funcionalidades que serão removidas
- **Removido**: Funcionalidades removidas
- **Corrigido**: Correções de bugs
- **Segurança**: Correções de vulnerabilidades

---

**Última atualização:** 2026-01-15
**Versão atual:** 0.1.0
**Status:** Versão inicial de desenvolvimento
