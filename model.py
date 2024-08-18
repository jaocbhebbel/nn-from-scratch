import numpy as np



# he initialization is used to assign weights 
# in a ReLU model random values.
# it prevents the ReLU death problem
# the np function called here returns a value in a normal distribution
# the parameters define the distribution to have a mean of 0, and variance sqrt(2/n)



def he_initialization(number_of_neurons_in):
    return np.random.normal(0, np.sqrt(2/n))


# relu is the activation function 
# chosen for this model because 
# of its computational simplicity

def relu(number):
    return max(0, number)



class Neuron:
    def __init__(self, isInput):
        self.isInput = isInput
        
        if isInput:
            self.weights = ""
            self.bias = ""
        else:
            self.weights = []
            self.bias = 0
        
        self.z = 0
        self.a = 0

    def activate(self):
        self.a = relu(self.a)

    def initializeWeights(self, n):
        for neuron in range(0, n):
            weights[neuron] = he_initialization(n)

    def adjustWeights(self, corrections_vector):
        for index in range(0, corrections_vector)
            weights[index] += corrections_vector[index]
    
    def adjustBias(self, correction):
        self.bias += correction


def main():
    
    model = []
    cost_table = []
    dataset = []

    # import dataset
    images = 
    labels = 

    for index in range(images.length()):
        dataset[index] = ( images[i], labels[i] )

    minibatches = randomize_dataset(dataset)
    
    for batch in range(0, minibatches.length()):
        cost_table[batch] = []

    for parameter in range(0, 784):
        inputLayer[parameter] = Neuron(isInput = True)

    hidden_layer = []
    for value in range(0, 16):
        hidden_layer[value] = Neuron(isInput = False)

    output_layer = []
    for probability in range(0, 10):
        output_layer[probability] = Neuron(isInput = False)

    model = [ input_layer,
             hidden_layer,
             hidden_layer,
             output_layer ]
    
    # initializing the model + cost table
    for layer in range(1, Model.length()):
        for neuron in Model[layer]:
            neuron.initializeWeights(Model[layer - 1].length())
            cost_table[0].insert( (0, []) ) 
    


    # now its time to take in some data
    # divide the data into mini-batches
    
    for minibatch in minibatches:
        for data in minibatch:
            
            # puts data in every input parameter
            for pixel in range (0, data[0].length()):
                model[0][pixel].setZ(data[0][pixel])
            
            for layer in range(1, model.length()):
                for neuron in layer:


        # feed data in



