import multiprocessing
import pandas as pd
from lag_llama_univariate_data import run_lag_llama_univariate  

if __name__ == "__main__":
    try:
        ''' 
        # INFO: RUN LAG LLAMA
        ''' 
        
        # Verifica se o método de início já foi definido para evitar erro
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

        # Cria um objeto de trava
        log_lock = multiprocessing.Lock()

        # Carrega o conjunto de dados combinado
        all_data = pd.read_csv('../database/combined_data.csv', sep=";")

        # Inicializa um dicionário para armazenar produtos para cada estado
        state_product_dict = {}

        # Itera sobre estados únicos
        for state in all_data['state'].unique():
            # Filtra produtos correspondentes a este estado
            products = all_data[all_data['state'] == state]['product'].unique()
            # Adiciona ao dicionário
            state_product_dict[state] = list(products)

        # Loop para cada estado e seus produtos
        for state, products in state_product_dict.items():
            for product in products:
                print(f"========== State: {state}, product: {product} ==========")

                # Filtra dados para o estado e produto atuais
                data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == product)]

                thread = multiprocessing.Process(target=run_lag_llama_univariate, args=(state, product))
                thread.start()
                thread.join() 

    except Exception as e:
        print("An error occurred:", e)
