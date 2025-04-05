## Pré-requisitos

* **Python:** Certifique-se de ter o Python instalado em seu sistema (versão compatível com o projeto).
* **uv:** A ferramenta `uv` deve estar instalada. Você pode instalá-la seguindo as instruções em sua [documentação oficial](https://docs.astral.sh/uv/).

## Sincronizando o Ambiente com uv astral

A ferramenta `uv astral` é utilizada para criar um ambiente virtual isolado e instalar as dependências.

1. **Navegue até o diretório do projeto:** Abra seu terminal e navegue até a pasta raiz do seu projeto onde o arquivo `requirements.txt` está localizado.

2. **Crie e sincronize o ambiente:** Execute o seguinte comando para criar (se não existir) e sincronizar o ambiente virtual com as dependências especificadas:

```bash
uv sync
```

## Executando o Arquivo `main.py`

```bash
uv run main.py
```
