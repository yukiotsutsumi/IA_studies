from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Implantar o gerador de números aleatórios, então ele gera os mesmos números
        # toda vez que o programa roda.
        random.seed(1)

        # Modelamos apenas um neurônio, com três conexões de entrada e uma conexão de saída.
        # Nós damos pesos aleatórios à uma matriz 3x1, com valores na faixa de -1 e 1, inclusive 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # A função Sigmoid, que descreve uma curva em forma de S.
    # Passamos a soma das entradas através desta função para normalizá-las entre 0 e 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # O derivativo da função de Sigmoid.
    # Esse é o gradiente da curva de Sigmoid.
    # Indica quão confiante somos sobre o peso existente.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Nós treinamos nossa rede neural por um processo de tentativa e erro, e ajustamos os pesos sinapticos toda vez.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Passamos o treinamento para a nossa rede neural.(com apenas um neurônio)
            output = self.think(training_set_inputs)

            # Calcula o erro (A diferença entre os dados de saída que queremos e os dados de saída 'previstos').
            error = training_set_outputs - output

            # Multiplicamos o erro pelo dado de entrada e pelo gradiente da curva de Sigmoid
            # Isso significa que pesos menos parecidos são mais ajustados
            # Isso significa que os dados de entrada que forem 0 não causarão problemas no processo.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Ajustamos os pesos
            self.synaptic_weights += adjustment

    # Instituimos um pensaento para nossa rede neural
    def think(self, inputs):
        # Passamos os dados para nossa rede neural (nosso único neurônio).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Inicializamos o pensamento do único neurônio
    neural_network = NeuralNetwork()

    print ("Pesos iniciais sinapticos aleatórios: ")
    print (neural_network.synaptic_weights)

    # O campo de treinamento. Temos 4 exemplos, cada um consiste em 3 valores de entrada e 1 valor de saída.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Treinamos a rede neural pelo nosso campo de treinamento
    # Fazemos isso 10000 vezes e ajustamos aos poucos o resultado!
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("Novos pesos sinapticos após o treinamento: ")
    print (neural_network.synaptic_weights)

    # Testamos a nossa rede neural em uma nova situação!
    print ("Considerando a nova situação: [1, 0, 0] -> ?: ")
print (neural_network.think(array([1, 0, 0])))

#---TUDO NO PROMPT DE COMANDO---
#cd - entra pasta
#cd.. - retorna para pasta passada
#python -m pip install numpy - instalar numpy na pasta que contém o python.exe
#python [nome do programa].py - executa o programa