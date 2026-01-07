---
name: Roadmap Multi-Point Statistics
overview: Roadmap abrangente para melhorias de código e extensão de funcionalidades do repositório de multi-point statistics para geração de meios porosos sintéticos, focando em simulações mais fidedignas e métricas de propriedades efetivas das rochas.
todos:
  - id: deps_management
    content: Criar requirements.txt com todas as dependências documentadas e versões testadas
    status: pending
  - id: refactor_utils
    content: "Refatorar scripts/utils.py em módulos organizados: visualization/plotting.py, data_processing/conversion.py, data_processing/hard_soft_data.py"
    status: pending
  - id: config_system
    content: Criar sistema de configuração em scripts/config/ com classes Config e suporte a YAML/JSON
    status: pending
  - id: metrics_expansion
    content: "Expandir scripts/metrics/ separando cálculo (compute.py) de visualização (display.py) e adicionar métricas: tortuosidade, distribuição de poros, fator de forma"
    status: pending
    dependencies:
      - refactor_utils
  - id: validation_module
    content: "Criar scripts/metrics/validation.py com funções de validação estatística: variogramas, entropia, SSIM, relatórios comparativos"
    status: pending
    dependencies:
      - metrics_expansion
  - id: evaluation_pipeline
    content: Implementar scripts/pipelines/evaluation_pipeline.py para avaliação automatizada de múltiplas realizações com relatórios CSV/HTML
    status: pending
    dependencies:
      - validation_module
  - id: mps_wrapper
    content: Criar scripts/core/simulator.py com classe MPSSimulator para encapsular configuração e execução de simulações
    status: pending
    dependencies:
      - config_system
  - id: parameter_optimization
    content: Implementar scripts/optimization/parameter_tuning.py para otimização de parâmetros MPS usando grid search
    status: pending
    dependencies:
      - evaluation_pipeline
      - mps_wrapper
  - id: ti_management
    content: Expandir gestão de training images em scripts/data_processing/ti_management.py com análise estatística e comparação de TIs
    status: pending
    dependencies:
      - refactor_utils
  - id: test_framework
    content: Criar estrutura de testes em tests/ com pytest, testes unitários de funções críticas e testes de integração de pipelines
    status: pending
    dependencies:
      - refactor_utils
      - metrics_expansion
  - id: documentation
    content: Expandir README.md com instalação, exemplos, estrutura do projeto e adicionar docstrings Google/NumPy em todas as funções
    status: pending
    dependencies:
      - refactor_utils
  - id: logging_system
    content: Implementar sistema de logging estruturado em scripts/utils/logging_config.py com níveis apropriados e progresso de simulações
    status: pending
    dependencies:
      - mps_wrapper
  - id: performance_optimization
    content: "Otimizar processamento: paralelização de métricas, cache de resultados, profiling de gargalos"
    status: pending
    dependencies:
      - metrics_expansion
  - id: simple_api
    content: Criar scripts/api/simple_api.py com funções de alto nível (run_simulation, evaluate_simulations) e CLI básica
    status: pending
    dependencies:
      - evaluation_pipeline
      - mps_wrapper
---

# Roadmap: Multi-Point Statistics para Meios Porosos

## Visão Geral do Repositório

O repositório implementa modelos de **Multi-Point Statistics (MPS)** usando a biblioteca `mpslib` para geração de meios porosos sintéticos em escala milimétrica/micrométrica. O código atual fornece:

- **Simulações MPS**: Usando algoritmos (genesim, snesim) através do mpslib
- **Processamento de dados**: Conversão entre formatos (.npy, .dat), hard/soft data
- **Visualização**: Plotting 2D/3D usando matplotlib e PyVista
- **Métricas básicas**: Porosidade, permeabilidade, área superficial, número de Euler
- **Patches de compatibilidade**: Para PyVista e NumPy (versões recentes)

## Fundamentos Teóricos

### Multi-Point Statistics (Strebelle, 2002)

O trabalho seminal de **Strebelle (2002)** estabelece as bases teóricas das estatísticas de múltiplos pontos para simulação condicional de estruturas geológicas complexas. Os conceitos fundamentais incluem:

1. **Limitações das Estatísticas de Dois Pontos**: Variogramas tradicionais capturam apenas correlações lineares entre pares de pontos, sendo inadequados para representar padrões curvilíneos complexos (ex: canais de areia sinuosos, estruturas geométricas não-lineares).

2. **Imagens de Treinamento (Training Images)**: As TIs são representações qualitativas dos padrões esperados de heterogeneidade geológica. Elas permitem inferir estatísticas de múltiplos pontos necessárias para a simulação, capturando padrões espaciais que não podem ser representados por variogramas.

3. **Algoritmo SNESIM (Single Normal Equation Simulation)**: O método utiliza uma estrutura de árvore de busca para armazenar padrões encontrados na TI, permitindo simulação sequencial eficiente baseada em padrões condicionais de múltiplos pontos.

4. **Simulação Condicional**: O método incorpora dados condicionais (hard data) para garantir que as realizações honrem informações conhecidas, mantendo ao mesmo tempo a reprodução dos padrões estatísticos da TI.

**Implicações para o Projeto**:

- A qualidade da TI é crítica para o sucesso das simulações
- A validação deve verificar a reprodução de padrões espaciais (não apenas estatísticas de dois pontos)
- Parâmetros como tamanho do template e número de pontos condicionais afetam a fidelidade à TI

### Reconstrução de Espaços Porosos (Blunt, 2005)

**Blunt (2005)** aplica MPS especificamente para reconstrução de meios porosos, focando em:

1. **Reconstrução 3D a partir de Imagens 2D**: O trabalho demonstra como usar MPS para reconstruir estruturas porosas tridimensionais a partir de imagens bidimensionais (como seções de rochas), capturando a continuidade espacial dos poros.

2. **Validação de Propriedades Efetivas**: O artigo enfatiza a importância de validar reconstruções comparando propriedades efetivas calculadas (porosidade, permeabilidade) com valores medidos experimentalmente ou com imagens 3D de referência.

3. **Métricas de Validação**: Além de métricas geométricas (porosidade), propriedades de transporte (permeabilidade) são fundamentais para validar a fidelidade das reconstruções. A estrutura interna do espaço poroso afeta diretamente propriedades como condutividade hidráulica e difusividade.

4. **Heterogeneidades e Propriedades Efetivas**: Heterogeneidades sutis no meio poroso podem impedir a obtenção de parâmetros efetivos únicos, destacando a necessidade de múltiplas realizações e análise estatística robusta.

**Implicações para o Projeto**:

- Validação deve incluir tanto métricas geométricas quanto propriedades efetivas de transporte
- Comparação com dados experimentais ou imagens de referência é essencial
- Múltiplas realizações devem ser analisadas estatisticamente para capturar variabilidade
- Métricas de conectividade e tortuosidade são críticas para propriedades de transporte

### Relação entre Fundamentos Teóricos e Objetivos do Projeto

Os fundamentos teóricos balizam as melhorias propostas:

1. **Fidelidade das Simulações**: Baseado em Strebelle, melhorias focam em otimização de parâmetros MPS e seleção/validação de TIs para garantir reprodução adequada de padrões complexos.

2. **Validação Abrangente**: Inspirado em Blunt, o projeto expande métricas de validação incluindo não apenas propriedades geométricas, mas também propriedades efetivas de transporte e conectividade.

3. **Análise Estatística Robusta**: Ambos os trabalhos enfatizam a importância de análise estatística adequada, justificando pipelines automatizados de avaliação e comparação com targets.

4. **Estrutura Espacial vs. Estatísticas de Dois Pontos**: O projeto deve implementar validações que vão além de variogramas, incluindo análise de padrões espaciais complexos e propriedades que dependem da estrutura completa do espaço poroso.

## Estrutura Atual

```
multi-point-statistics/
├── scripts/
│   ├── utils.py              # Funções utilitárias (plotting, conversão de dados)
│   ├── ti_saver.py           # Salvamento/carregamento de training images
│   ├── metrics/
│   │   └── metrics_display.py  # Cálculo de métricas usando poregen.features
│   └── patches/              # Patches de compatibilidade
├── notebooks/
│   ├── mps_examples/         # Exemplos básicos do mpslib
│   └── mps_scalings/         # Experimentos de escalonamento
└── articles/                 # Artigos científicos de referência
```

## Roadmap de Melhorias

### Fase 1: Fundamentos e Estrutura (Prioridade Alta)

#### 1.1 Gestão de Dependências e Ambiente

- **Objetivo**: Garantir reprodutibilidade e fácil instalação
- **Ações**:
  - Criar `requirements.txt` com todas as dependências (mpslib, numpy, matplotlib, pyvista, poregen, torch, scipy, pandas, sklearn)
  - Criar `environment.yml` para Conda (opcional, mas recomendado)
  - Documentar versões testadas e compatibilidades
  - Arquivo: `requirements.txt` (novo)

#### 1.2 Refatoração e Organização de Código

- **Objetivo**: Melhorar manutenibilidade e reutilização
- **Ações**:
  - Modularizar `scripts/utils.py` (atualmente ~750 linhas):
    - `scripts/visualization/plotting.py`: Funções de plot (2D/3D)
    - `scripts/data_processing/conversion.py`: Conversão de formatos
    - `scripts/data_processing/hard_soft_data.py`: Manipulação de dados condicionais
    - `scripts/utils.py`: Manter apenas funções de conveniência que importam das subpastas
  - Criar classe wrapper `MPSSimulator` em `scripts/core/simulator.py` para encapsular configuração e execução de simulações
  - Arquivos: Múltiplos novos em `scripts/`

#### 1.3 Configuração e Parâmetros

- **Objetivo**: Facilitar configuração e experimentação
- **Ações**:
  - Criar `scripts/config/default_config.py` com parâmetros padrão
  - Implementar `Config` class usando dataclasses ou pydantic para validação
  - Suporte a arquivos YAML/JSON para configurações customizadas
  - Arquivo: `scripts/config/` (novo)

### Fase 2: Métricas e Validação (Prioridade Alta)

**Base Teórica**: Esta fase implementa diretamente os conceitos de validação apresentados por Blunt (2005) e os princípios de avaliação de fidelidade de Strebelle (2002).

#### 2.1 Expansão do Módulo de Métricas

- **Objetivo**: Ampliar métricas de propriedades efetivas das rochas (conforme Blunt, 2005)
- **Base Teórica**: Blunt enfatiza que validação adequada requer tanto métricas geométricas quanto propriedades efetivas de transporte
- **Ações**:
  - Reorganizar `scripts/metrics/`:
    - `metrics/compute.py`: Funções de cálculo (separar de display)
    - `metrics/display.py`: Visualização e comparação
    - `metrics/validation.py`: Validação estatística (comparação com target)
  - Adicionar métricas adicionais mantendo base em `poregen.features`:
    - **Propriedades geométricas** (já existem: porosidade, área superficial):
      - Distribuição de tamanho de poros (Pore Size Distribution)
      - Fator de forma (aspect ratio dos poros)
    - **Propriedades de conectividade e transporte** (conforme Blunt):
      - Tortuosidade (métrica crítica para transporte)
      - Coordenação (connectivity index - número de conexões por poro)
      - Distribuição de comprimento de gargalos (throat size distribution)
      - Conectividade efetiva (fração de poros conectados)
  - Criar classe `MetricsCalculator` para cálculo em lote com cache
  - Arquivos: Refatoração de `scripts/metrics/metrics_display.py`

#### 2.2 Validação Estatística de Simulações

- **Objetivo**: Avaliar fidelidade das simulações ao target (conforme Strebelle, 2002)
- **Base Teórica**: Strebelle destaca que validação adequada deve verificar reprodução de padrões espaciais complexos, não apenas estatísticas de dois pontos
- **Ações**:
  - Implementar funções de comparação estatística hierárquica:
    - **Estatísticas de dois pontos** (baseline, mas insuficiente por si só):
      - Two-point correlation function (já existe parcialmente em `utils.py`)
      - Variogramas direcionais (anisotropia)
    - **Estatísticas de múltiplos pontos** (conforme Strebelle):
      - Análise de padrões espaciais (spatial pattern analysis)
      - Distribuição de configurações locais (template matching statistics)
      - Entropia e informação mútua (medida de complexidade e similaridade)
    - **Métricas de similaridade estrutural**:
      - SSIM (Structural Similarity Index) - captura padrões espaciais
      - MSE, MAE (baseline)
      - Métricas baseadas em histogramas de padrões locais
  - Comparação de propriedades efetivas com target (conforme Blunt)
  - Criar relatórios automáticos de validação com múltiplos níveis de análise
  - Arquivo: `scripts/metrics/validation.py` (novo)

#### 2.3 Pipeline de Avaliação Automatizado

- **Objetivo**: Avaliar múltiplas realizações automaticamente (essencial conforme Blunt para análise estatística robusta)
- **Base Teórica**: Blunt enfatiza que heterogeneidades sutis requerem análise estatística de múltiplas realizações para capturar variabilidade adequada
- **Ações**:
  - Criar `scripts/pipelines/evaluation_pipeline.py`:
    - Carregar simulações e target
    - Calcular todas as métricas (geométricas e de transporte)
    - Análise estatística de ensemble:
      - Distribuições de métricas através das realizações
      - Comparação com intervalo de confiança do target
      - Identificação de realizações outliers
      - Análise de convergência (como métricas variam com número de realizações)
    - Gerar relatórios comparativos (CSV, HTML) com:
      - Estatísticas descritivas (média, desvio padrão, quartis)
      - Testes estatísticos (t-test, Kolmogorov-Smirnov) comparando realizações com target
      - Visualizações de distribuições e comparações
    - Visualizações automatizadas:
      - Histogramas comparativos
      - Box plots de métricas
      - Gráficos de correlação entre métricas
  - Integrar com sistema de logging
  - Arquivo: `scripts/pipelines/` (novo)

### Fase 3: Melhorias de Simulação (Prioridade Média-Alta)

#### 3.1 Otimização de Parâmetros MPS

- **Objetivo**: Facilitar ajuste de parâmetros para simulações mais fidedignas (conforme Strebelle, 2002)
- **Base Teórica**: Strebelle discute como parâmetros como tamanho do template e número de pontos condicionais afetam diretamente a capacidade de reproduzir padrões complexos da TI
- **Ações**:
  - Criar `scripts/optimization/parameter_tuning.py`:
    - **Grid search para parâmetros principais**:
      - `n_cond`: Número de pontos condicionais (afeta fidelidade vs. variabilidade)
      - `n_max_cpdf_count`: Para genesim - afeta qualidade da busca de padrões
      - Tamanho do template (affects capacidade de capturar padrões grandes)
      - Método de busca (sequential path, random path)
    - **Otimização baseada em métricas de validação**:
      - Função objetivo combinando múltiplas métricas (porosidade, permeabilidade, correlação espacial)
      - Pesos configuráveis para diferentes tipos de métricas
      - Validação cruzada usando múltiplas realizações
    - **Análise de sensibilidade**:
      - Identificar parâmetros mais críticos
      - Trade-offs entre diferentes objetivos (velocidade vs. qualidade)
    - **Suporte a otimização bayesiana** (opcional, usando scikit-optimize):
      - Mais eficiente que grid search para espaços de parâmetros grandes
      - Exploração vs. exploração balanceada
  - Documentar parâmetros críticos e seus efeitos teóricos (baseado em literatura MPS)
  - Integrar com pipeline de avaliação para feedback automatizado
  - Arquivo: `scripts/optimization/` (novo)

#### 3.2 Suporte Melhorado a Hard e Soft Data

- **Objetivo**: Facilitar incorporação de dados condicionais
- **Ações**:
  - Expandir funções em `scripts/data_processing/hard_soft_data.py`:
    - Validação de dados condicionais
    - Visualização de distribuição espacial de dados
    - Ferramentas para amostragem estratégica de hard data
  - Criar exemplos documentados de uso avançado
  - Melhorar `npy_to_hard_data()` em `utils.py` (já existe, mas pode ser expandido)

#### 3.3 Gestão de Training Images (TI)

- **Objetivo**: Facilitar preparação e seleção de TIs (crítico conforme Strebelle, 2002)
- **Base Teórica**: Strebelle estabelece que a qualidade e adequação da TI é fundamental para o sucesso das simulações. A TI deve representar adequadamente os padrões espaciais desejados.
- **Ações**:
  - Expandir `scripts/ti_saver.py` em `scripts/data_processing/ti_management.py`:
    - **Análise estatística de TIs**:
      - Porosidade e distribuição de valores
      - Análise de padrões espaciais (orientação, continuidade)
      - Análise de anisotropia
      - Estatísticas de múltiplos pontos da própria TI (para referência)
    - **Validação de adequação da TI**:
      - Comparação da TI com target (se disponível)
      - Verificação de padrões representativos
      - Análise de complexidade (entropia)
    - **Comparação de múltiplas TIs**:
      - Identificar qual TI produz melhores resultados
      - Análise de trade-offs (complexidade vs. fidelidade)
    - **Ferramentas para pré-processamento**:
      - Normalização e limpeza
      - Filtragem de ruído
      - Ajuste de resolução/escala
      - Extração de sub-regiões representativas
    - **Catálogo de TIs com metadados**:
      - Propriedades estatísticas
      - Dimensões e resolução
      - Origem (experimental, sintética, literatura)
      - Parâmetros de simulação recomendados
  - Integrar análise de TI no pipeline de validação
  - Arquivo: `scripts/data_processing/ti_management.py` (novo)

### Fase 4: Qualidade de Código e Testes (Prioridade Média)

#### 4.1 Testes Unitários

- **Objetivo**: Garantir robustez do código
- **Ações**:
  - Criar `tests/` com estrutura:
    - `tests/unit/`: Testes de funções individuais
    - `tests/integration/`: Testes de pipelines completos
  - Usar pytest como framework
  - Cobertura inicial: funções críticas de processamento e métricas
  - Arquivo: `tests/` (novo), `pytest.ini` (novo)

#### 4.2 Documentação

- **Objetivo**: Facilitar uso e contribuição
- **Ações**:
  - Expandir `README.md` com:
    - Instalação detalhada
    - Exemplos de uso principais
    - Estrutura do projeto
  - Adicionar docstrings em todas as funções (usar formato Google/NumPy)
  - Criar `docs/` com documentação Sphinx (opcional, mas recomendado)
  - Tutorials em notebooks: `notebooks/tutorials/`
  - Arquivo: `README.md` (expandir), `docs/` (novo)

#### 4.3 Logging e Debugging

- **Objetivo**: Facilitar desenvolvimento e troubleshooting
- **Ações**:
  - Implementar sistema de logging estruturado
  - Criar `scripts/utils/logging_config.py`
  - Níveis apropriados (DEBUG, INFO, WARNING, ERROR)
  - Integrar com visualizações de progresso para simulações longas
  - Arquivo: `scripts/utils/logging_config.py` (novo)

### Fase 5: Performance e Escalabilidade (Prioridade Média)

#### 5.1 Otimização de Processamento

- **Objetivo**: Acelerar simulações e processamento
- **Ações**:
  - Paralelização de cálculo de métricas (usando multiprocessing/joblib)
  - Otimização de operações NumPy (vectorização onde possível)
  - Cache de resultados intermediários
  - Profiling e identificação de gargalos
  - Arquivo: Modificações em `scripts/metrics/compute.py`

#### 5.2 Suporte a Dados Grandes

- **Objetivo**: Lidar com simulações 3D grandes
- **Ações**:
  - Suporte a processamento em chunks
  - Integração com Dask para arrays grandes (opcional)
  - Ferramentas para downsampling inteligente
  - Compressão de resultados (HDF5, zarr)

### Fase 6: Extensões Avançadas (Prioridade Baixa-Média)

#### 6.1 Interface de Alto Nível

- **Objetivo**: Simplificar uso para usuários finais
- **Ações**:
  - Criar `scripts/api/simple_api.py` com funções de alto nível:
    - `run_simulation()`: Uma função que faz tudo
    - `evaluate_simulations()`: Avaliação completa
  - CLI básica usando Click ou argparse
  - Arquivo: `scripts/api/` (novo), `scripts/cli.py` (novo)

#### 6.2 Visualizações Avançadas

- **Objetivo**: Melhorar análise visual
- **Ações**:
  - Dashboard interativo (usando Plotly Dash ou Streamlit)
  - Visualização de distribuições de métricas
  - Animações de simulações
  - Comparação lado-a-lado avançada

#### 6.3 Integração com Outras Ferramentas

- **Objetivo**: Expandir capacidades
- **Ações**:
  - Integração com bibliotecas de análise de rochas (se disponíveis)
  - Exportação para formatos padrão (VTK, HDF5)
  - Suporte a workflows automatizados (prefect, luigi)

## Priorização Sugerida (Ordem de Implementação)

### Sprint 1-2 (Fundamentos)

1. Gestão de dependências (`requirements.txt`)
2. Refatoração básica de `utils.py` (modularização)
3. Expansão do módulo de métricas (estrutura + métricas básicas adicionais)
4. Configuração básica (`config/`)

### Sprint 3-4 (Validação)

5. Validação estatística completa
6. Pipeline de avaliação automatizado
7. Testes unitários básicos
8. Melhorias na documentação (`README.md`)

### Sprint 5-6 (Otimização)

9. Otimização de parâmetros
10. Melhorias de performance
11. Sistema de logging
12. Testes de integração

### Sprint 7+ (Extensões)

13. Interface de alto nível
14. Visualizações avançadas
15. Outras extensões conforme necessidade

## Arquivos Principais a Criar/Modificar

### Novos Módulos

- `requirements.txt`
- `scripts/core/simulator.py`
- `scripts/config/default_config.py`
- `scripts/metrics/compute.py`
- `scripts/metrics/validation.py`
- `scripts/pipelines/evaluation_pipeline.py`
- `scripts/data_processing/` (vários arquivos)
- `scripts/visualization/plotting.py`
- `tests/` (estrutura completa)

### Arquivos a Refatorar

- `scripts/utils.py` → Dividir em múltiplos módulos
- `scripts/metrics/metrics_display.py` → Separar cálculo de visualização
- `README.md` → Expandir significativamente

## Boas Práticas Baseadas nos Fundamentos Teóricos

### Seleção e Preparação de Training Images (Strebelle, 2002)

1. **Representatividade**: A TI deve representar adequadamente os padrões espaciais desejados. Para meios porosos, deve capturar heterogeneidades relevantes da estrutura porosa.

2. **Tamanho Adequado**: A TI deve ser suficientemente grande para conter múltiplas ocorrências dos padrões principais, permitindo inferência estatística robusta.

3. **Resolução**: A resolução da TI deve ser compatível com a escala de heterogeneidades que se deseja reproduzir.

4. **Validação Prévia**: Antes de usar uma TI em simulações, verificar suas propriedades estatísticas e compará-las com o target (se disponível).

### Validação de Simulações (Blunt, 2005)

1. **Múltiplas Métricas**: Usar tanto métricas geométricas (porosidade) quanto propriedades efetivas de transporte (permeabilidade) para validação completa.

2. **Múltiplas Realizações**: Analisar estatisticamente um conjunto de realizações para capturar variabilidade e identificar outliers.

3. **Comparação com Referência**: Sempre que possível, comparar simulações com dados experimentais ou imagens 3D de referência para validação quantitativa.

4. **Análise de Conectividade**: Propriedades de transporte dependem criticamente da conectividade do espaço poroso. Métricas como tortuosidade são essenciais.

5. **Validação Espacial**: Além de métricas globais, verificar a reprodução de padrões espaciais locais (conforme Strebelle).

### Ajuste de Parâmetros (Strebelle, 2002)

1. **Template Size**: Deve ser grande o suficiente para capturar padrões principais, mas não excessivamente grande para manter eficiência.

2. **Número de Condicionais**: Balancear entre fidelidade (mais condicionais) e variabilidade entre realizações (menos condicionais).

3. **Validação Iterativa**: Ajustar parâmetros baseado em resultados de validação, iterativamente refinando até obter simulações satisfatórias.

## Considerações Técnicas

- **Compatibilidade**: Manter suporte a versões recentes de Python (3.8+)
- **Dependências externas**: `mpslib`, `poregen.features` são críticas
- **Patches**: Manter sistema de patches para compatibilidade
- **Notebooks**: Manter notebooks como exemplos, mas código principal em módulos Python
- **Reprodutibilidade**: Garantir que simulações sejam reproduzíveis (seeds, versionamento de parâmetros)
- **Documentação Teórica**: Incluir referências aos artigos fundamentais na documentação de funções críticas

## Métricas de Sucesso

- Cobertura de testes > 70% para módulos críticos
- Documentação completa de todas as funções públicas com referências teóricas quando aplicável
- Tempo de simulação reduzido em 20-30% (onde aplicável)
- Facilidade de uso: novos usuários conseguem rodar simulação completa em < 30 minutos
- Métricas de validação robustas e bem documentadas:
  - Validação geométrica (porosidade, distribuição de tamanho de poros)
  - Validação de transporte (permeabilidade, tortuosidade)
  - Validação espacial (reprodução de padrões da TI)
- Fidelidade das simulações: propriedades efetivas das realizações dentro de 10-15% dos valores do target (quando disponível)
- Análise estatística: relatórios automatizados incluindo análise de ensemble, intervalos de confiança e testes estatísticos

## Referências Principais

- **Strebelle, S. (2002)**: "Conditional simulation of complex geological structures using multiple-point statistics" - Fundamentos teóricos de MPS e algoritmo SNESIM
- **Blunt, M.J. (2005)**: "Pore space reconstruction using multiple-point statistics" - Aplicação de MPS para meios porosos e validação de propriedades efetivas