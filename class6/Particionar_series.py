#-*- coding: utf-8 -*-
import numpy as np
import copy

class Particionar_series:
    def __init__(self, serie, divisao, janela):
        '''
        classe para manipular a serie temporal
        :param serie: vetor com a serie temporal a ser manipulada 
        :param divisao: lista com porcentagens, da seguinte forma [pct_treinamento, pct_validacao, pct_teste]
        :param janela: quantidade de lags usados para modelar os padroes de entrada da Rede
        '''
        
        self.min = 0
        self.max = 0
        self.serie = serie
        self.serie = self.Normalizar(serie)
        self.janela = janela
        
        if(len(divisao) != 3):
            print("Erro no tamanho da particao!")
            
        self.pct_train = divisao[0]
        self.pct_val = divisao[1]
        self.pct_test = divisao[2]
       
        self.tam_train = 0
        self.tam_val = 0
        self.tam_test = 0
        
    def Part_train(self):
        '''
        metodo que retorna somente a parte de treinamento da serie temporal
        :return: retorna uma lista com: [entrada_train, saida_train]
        '''
        
        self.tam_train = self.pct_train * len(self.serie)
        self.tam_train = int(round(self.tam_train))
        
        serie = self.serie[:self.tam_train]
        
        [entrada_train, saida_train] = self.Janela_tempo(serie)
        
        entrada_train = np.asarray(entrada_train)
        saida_train = np.asarray(saida_train)
        
        return entrada_train, saida_train
    
    def Part_val(self):
        '''
        metodo que retorna somente a parte de validacao da serie temporal
        :return: retorna uma lista com: [entrada_val, saida_val]
        '''
        
        self.tam_val = self.pct_val * len(self.serie)
        self.tam_val = int(round(self.tam_val))
        
        serie = self.serie[self.tam_train:self.tam_train + self.tam_val]
        
        [entrada_val, saida_val] = self.Janela_tempo(serie)
        
        entrada_val = np.asarray(entrada_val)
        saida_val = np.asarray(saida_val)
        
        return entrada_val, saida_val
    
    def Part_test(self):
        '''
        metodo que retorna somente a parte de teste da serie temporal
        :return: retorna uma lista com: [entrada_teste, saida_teste]
        '''
          
        self.tam_test = self.pct_test * len(self.serie)
        self.tam_test = int(round(self.tam_test))
        
        serie = self.serie[self.tam_train + self.tam_val: ]
        
        [entrada_teste, saida_teste] = self.Janela_tempo(serie)
        
        entrada_teste = np.asarray(entrada_teste)
        saida_teste = np.asarray(saida_teste)
        
        return entrada_teste, saida_teste
    
    def Janela_tempo(self, serie):
        '''
        metodo que transforma um vetor em uma matriz com os dados de entrada e um vetor com as respectivas saidas 
        :param serie: serie temporal que sera remodelada
        :return: retorna duas variaveis, uma matriz com os dados de entrada e um vetor com os dados de saida: matriz_entrada, vetor_saida
        '''
        
        tamanho_matriz = len(serie) - self.janela 
        
        matriz_entrada = []
        for i in range(tamanho_matriz):
            matriz_entrada.append([0.0] * self.janela)
        
        vetor_saida = []
        for i in range(len(matriz_entrada)):
            matriz_entrada[i] = serie[i:i+self.janela]
            vetor_saida.append(serie[i+self.janela])
            
        return matriz_entrada, vetor_saida
    
    def Normalizar(self, serie):
        '''
        metodo que normaliza a serie temporal em um intervalo de [0, 1] 
        :param serie: serie temporal que sera remodelada
        :return: retorna a serie normalizada 
        '''
        
        self.min = copy.deepcopy(np.min(serie))
        self.max = copy.deepcopy(np.max(serie))
        
        serie_norm = []
        for e in serie:
            valor = (e - self.min)/(self.max - self.min)
            serie_norm.append(valor)
        
        return serie_norm  
     
    def Desnormalizar(self, serie):
        '''
        metodo que retira a normalizacao da serie e a coloca em escala original 
        :param serie: serie temporal que sera remodelada
        :return: retorna a serie na escala original  
        '''

        serie_norm = []
        for e in serie:
            valor = e * (self.max - self.min) + self.min
            serie_norm.append(valor)

        return serie_norm  
        
def main():
    
    # exemplo de uma serie
    serie = [1, 2, 4, 7 ,11, 25, 30, 1, 2, 4, 7 ,11, 25, 30, 1, 2, 4, 7 ,11, 25, 30, 1, 2, 4, 7 ,11, 25, 30, 5]
    
    # exemplo de divisao
    divisao = [0.65, 0.15, 0.2]
    
    # instanciando a classe
    particao = Particionar_series(serie, divisao, 4)
    
    # printando algumas variaveis
    print(particao.serie)
    print(particao.min)
    print(particao.max)
    print(particao.Desnormalizar(particao.serie))
    print(serie)
    
    # como particionar o dataset
    [train_entrada, train_saida] = particao.Part_train()
    [val_entrada, val_saida] = particao.Part_val()
    [test_entrada, test_saida] = particao.Part_test()
    
    print("train_entrada:", len(train_entrada))
    print("train_saida:", len(train_saida))
    print("val_entrada:", len(val_entrada))
    print("val_saida:", len(val_saida))
    print("test_entrada:", len(test_entrada))
    print("test_saida:", len(test_saida))
    
if __name__ == "__main__":
    main()


