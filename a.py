from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Carregar os dados do arquivo CSV para um DataFrame
df = pd.read_csv('bread_basket.csv', sep=';')

# Agrupar os itens por transação
grouped_df = df.groupby('Transaction')['Item'].apply(list).reset_index()

# Converter os dados para o formato esperado pelo Apriori
te = TransactionEncoder() # Transformar os dados em um array com itens únicos
te_ary = te.fit_transform(grouped_df['Item']) # ransforma em um valor booleano
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)  # Cada coluna representa um item e cada linha uma transação

# Executar o algoritmo Apriori com tamanho máximo das regras definido para 3
frequent_itemsets = apriori(df_encoded, min_support=0.0011, use_colnames=True)
# min_support: valor mínimo de suporte para gerar os itens frequentes
# apriori: função de regras de associação do mlxtend.frequent_patterns

# Gerar as regras de associação com confiança
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01) 
# min_threshold: valor mínimo de confiança para gerar as regras
# association_rules: função de regras de associação do mlxtend.frequent_patterns

# Filtrar as regras com três itens
rules_3_items = rules[rules['antecedents'].apply(lambda x: len(x) == 3)]
# apply(lambda x: len(x) == 3): regra de associação de três itens antecedentes

# Exibir as regras de associação com suporte e confiança
print(rules_3_items[['support', 'confidence', 'antecedents', 'consequents']])