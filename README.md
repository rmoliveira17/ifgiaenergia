# ifgiaenergia
Repositório destinado ao Trabalho final do Módulo 2 da Especialização em IA do IFG

* Executar e instalar dependências (requirements.txt)

```
pip install requirements.txt
```

Inicialmente, é possível executar os arquivos de treinamento e teste sem hiperparametrização (xgbooost_ok.py e rna_ok) pela IDE com os dados locais. Para os dados na nuvem (Snowflake), é necessário chave de API e autorização. Foi disponibilizado um conjunto em Excel para facilitar o acesso. Os arquivos com hiperparametrização via Optuna podem ser executados também inicialmente através da IDE. Para predição de uma cidade individual, num certo ano, se deve utilzar o seguinte comando:

* XGboost
```
python predicting_xgboost.py   --cidade "Goiânia"   --ano 2023   --arquivo "Modelo Final.xlsx"
```

* MLP
```
python predicting_RNA.py   --cidade "Goiânia"   --ano 2024   --arquivo "Modelo Final.xlsx"
```

As cidades disponíveis e período para previsão estão no arquivo "Modelo Final.xlsx". Os estudos com o Optuna podem ser acessados com o seguinte comando via dashboard:

* XGboost

```
 optuna-dashboard sqlite:///xgb_fv_optuna.db

```

* MLP

```
optuna-dashboard sqlite:///mlp_fv_optuna.db 
```

O mapa com resultado das previsões pode ser acessado os resultados através do comando no prompt:

```
python xgb_map_goias.py   --ano 2024   --arquivo "Modelo Final.xlsx"
```

