#-*- coding: utf-8 -*-

'''
Created on 20 de ago de 2017

@author: gusta
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.regression import mean_squared_error
from Particionar_series import Particionar_series

def Plotar_predicoes(serie, lags, predicao_treinamento, predicao_teste, mse_train, mse_test):
    '''
    método para plotar as previsoes sobre a serie real
    :param: serie: serie original
    :param: lags: quantidade de lags usadas no treinamento para ajustar a previsao
    :param: predicao_treinamento: predicao do conjunto de treinamento
    :param: predicao_teste: predicao do conjunto de teste
    '''
    
    # plotando a serie original
    plt.plot(serie, label = "Série original")
    
    
    # criando uma variavel para o plot do conjunto de treinamento
    pred_train = [None] * len(serie)
    # atribuindo o espaçamento dos lags de entrada
    inicio = lags
    # passando os dados para a variavel que será pltoada
    pred_train[inicio:] = predicao_treinamento
    # plotando a variavel
    string = "Previsão Treinamento - MSE: %.3f" % mse_train
    plt.plot(pred_train, label = string)
    
    
    # criando uma variavel para o plot do conjunto de teste
    pred_test = [None] * len(serie)
    # atribuindo o espaçamento dos lags de entrada
    inicio = inicio+lags+len(predicao_treinamento)
    # passando os dados para a variavel que será pltoada
    pred_test[inicio:] = predicao_teste
    # plotando a variavel
    string = "Previsão Teste - MSE: %.3f" % mse_test
    plt.plot(pred_test, label = string)
    
    # mostrando o grafico
    plt.legend()
    plt.show()
    
def Leitura_dados(caminho, excel = None, csv = None):
        '''
        Metodo para fazer a leitura dos dados
        :param caminho: caminho da base que sera importada
        :return: retorna a serie temporal que o caminho direciona
        '''
        #leitura da serie dinamica
        
        if(excel == True):
            print(caminho)
            stream = pd.read_excel(caminho, header = None)
            stream = stream[0]
            stream = stream.as_matrix()
            return stream
        
        elif(csv == True):
            print(caminho)
            stream = pd.read_csv(caminho, header = None)
            stream = stream[0]
            stream = stream.as_matrix()
            return stream
        
def Importar_series(numero):
    '''
    método para importar os datasets do projeto
    :param: numero: valor inteiro que diz qual o dataset referente.
    :return: retorna a serie importada
    '''
    
    '''
    Esses datasets podem ser encontrados no seguinte repositório: https://datamarket.com/data/list/?q=provider:tsdl
    '''
    
    pasta = 'series/'
    
    if(numero == 0):
        caminho = pasta+'annual-water-use-in-new-york-cit.csv'
        return Leitura_dados(caminho, csv=True)
    if(numero == 1):
        caminho = pasta+'exchange-rate-of-australian-doll.csv'
        return Leitura_dados(caminho, csv=True)
    if(numero == 2):
        caminho = pasta+'exchange-rate-twi-may-1970-aug-1.csv'
        return Leitura_dados(caminho, csv=True)
    if(numero == 3):
        caminho = pasta+'ibm-common-stock-closing-prices.csv'
        return Leitura_dados(caminho, csv=True)
    if(numero == 4):
        caminho = pasta+'quarterly-increase-in-stocks-non.csv'
        return Leitura_dados(caminho, csv=True)

def main():
    # importando o dataset
    serie = Importar_series(4)
    
    # definindo os lags de entrada
    lags = 4
    
    # instanciando a classe
    particao = Particionar_series(serie, [0.8, 0.2, 0], lags)
    
    # como particionar o dataset
    [train_entrada, train_saida] = particao.Part_train()
    [val_entrada, val_saida] = particao.Part_val()
    
    # instanciando um modelo de mlp
    rede = MLPRegressor(
        hidden_layer_sizes=(50,),  
        activation='relu', 
        solver='adam', 
        alpha=0.001, 
        batch_size='auto',
        learning_rate='constant', 
        learning_rate_init=0.001, 
        power_t=0.5, 
        max_iter=1000, 
        shuffle=True,
        random_state=9,
        tol=0.0001, 
        verbose=False, 
        warm_start=False, 
        momentum=0.9, 
        nesterovs_momentum=True,
        early_stopping=True, 
        validation_fraction=0.1, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-08)
    
    # treinando o modelo com base no conjunto de treinamento
    rede = rede.fit(train_entrada, train_saida)
    
    # computando a predicao para os dados de treinamento e teste
    previsao_treinamento = rede.predict(train_entrada)
    previsao_teste = rede.predict(val_entrada)
    
    '''
    # retirando a normalizacao dos dados
    train_saida = particao.Desnormalizar(train_saida)
    previsao_treinamento = particao.Desnormalizar(previsao_treinamento)
    val_saida = particao.Desnormalizar(val_saida)
    previsao_teste = particao.Desnormalizar(previsao_teste)
    particao.serie = particao.Desnormalizar(particao.serie)
    '''
    
    # printando o erro de treinamento e teste
    mse_train = mean_squared_error(train_saida, previsao_treinamento)
    print("Erro treinamento - MSE: ", mse_train)
    mse_test = mean_squared_error(val_saida, previsao_teste)
    print("Erro teste - MSE: ", mse_test)
    
    
    # plotando as previsoes
    Plotar_predicoes(particao.serie, lags, previsao_treinamento, previsao_teste, mse_train, mse_test)
    
if __name__ == "__main__":
    main()
