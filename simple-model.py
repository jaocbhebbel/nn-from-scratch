import numpy as np

def relu(number):
    return max(0, number)

def he_init(neuron_count_in_prev_layer):
    return np.random.normal(0, np.sqrt(2/n))

def softmax(x):
    
    probability_distribution = []
    e_j = sum(np.exp(x))

    for x_i in x:
        e_x = np.exp(x_i)
        probability = round((e_x / e_j), 4)
        probability_distribution.insert(probability)
    
    return probability_distribution



class Neuron:
    def __init__(self, isInput, isOutput):
        self.isInput = isInput
        self.isOutput = isOutput
        
        self.z = 0
        self.a = 0

        self.weights = []
        self.bias = 0
        self.probability = 0

    def initialize_weights(self, neuron_count):
        
        if self.isInput:
            return None

        for neuron in range(0, neuron_count):
            weights[neuron] = he_init(neuron_count)

    def activate(self):
        self.a = relu(self.z)




# non-class functions:



# returns a set of 700 random ints between -10 and 15
def makeDataset():
    rng = np.random.default_rng(12345)
    numbers = rng.ints(low=-10, high=15, size=700)
    return numbers





# breaks the dataset into minibatches for testing purposes
def makeMinibatches(array, batch_size):
    return [array[i:i + batch_size] for i in range(0, len(array), batch_size])





# starts the propagation process
# it fills in the input layer with values, 
# and sparks the recursive function

def propagate_forward(parameter, model):
    

    # if parameter is a vector and the model has a many input nodes
    # this would be a for loop to initialize every input node

    input_neuron = model[0][0]
    input_neuron.z = parameter
    input_neuron.activate();


    # defining the input layer is separate because it is not recursive
    # all other z values depend on previous activation values, 
    # which depend on activation values before them, etc etc


    # again skipping input layer; was already initialized
    for layer in range(1, len(model)):
        
        # propagation will be done by layer, and moving on when 
        # all neurons have their activated value
        for neuron in model[layer]:
            
            # represents the sigma term in z equation
            weight_activation_sum = 0
            
            # accesses neuron activations in the previous layer
            # previous layer neurons should always be defined
            for index in range(0, len(neuron.weights)):
                weight_activation_sum += neuron.weights[index] * model[layer-1][index].a
            
            # z = sum(a_i * w_i) + bias
            # a = f(z), in this model f is the ReLU function, defined at the top
            neuron.z = return_activated(layer, model) + neuron.bias
            neuron.activate()

    
    # at this point, the model has finished propagating. the last nodes have values,
    # to make sense of these, take them as a probability distribution that a certain
    # neuron is the "correct answer" ie the model's confidence that a neuron is right

    # the funny thing about this is, the computer doesn't know which is the "right" neuron,
    # we shape this concept by adjusting the weights to whatever construct we want.
    

    # to get the probability vector of these values, we'll use the softmax function

    probabilities = softmax(model[-1])

    return probabilities


def cost_function(probability_distribution, input_parameter):
    
    model_prob = probability_distribution
    model_exp = []

    input_cost = []


    # here we define our construct
    
    # let model_prob[0] be the probability the input parameter is greater than 5
    # let model_prob[1] be the probability the input parameter is equal to 5
    # let model_prob[2] be the probability the input parameter is less than 5
    



    if input_parameter < 5:
        model_exp[0] = 0
        model_exp[1] = 0
        model_exp[2] = 1
    elif input_parameter > 5:
        model_exp[0] = 1
        model_exp[1] = 0
        model_exp[2] = 0
    else:
        model_exp[0] = 0
        model_exp[1] = 1
        model_exp[2] = 2
    
    for index in range(0, 3):
        input_cost[index] = (model_prob[index] - model_exp[index])^2 

    return input_cost



def main():
    
    dataset = makeDataset()
    minibatches = makeMinibatches(dataset, batch_size=50)


    # holds the cost of the model as entries in a 2D array
    # rows are minibatches, cols are the input's cost, as an array of tuples
    # tuples represent a neuron's bias and weights adjustments (dC / dB and dC / dWz respectively)
    cost = []

    learning_rate = 0


    # model
    model = []

    input_layer = [ Neuron(isInput=True, isOutput=False) ]

    hidden_layer = [ Neuron(isInput=False, isOutput=False) for neuron in range(2) ]

    output_layer = [ Neuron(isInput=False, isOutput=True) for neuron in range(3) ]
    
    model = [ input_layer, # one neuron that holds the input value
             hidden_layer, # two neurons, each with 1 weight to the input layer and a bias term
             output_layer ] # three neurons, representing three outcomes. have weights + bias terms. apply softmax to get probability distribution of neuron k being the input value





    # initializing weights
    for layer in range(1, len(model)): #starts from 1 bc layer 0 has no weights
        for neuron in model[layer]:
            
            # the func arg is the number of neurons in the previous layer
            # the init method, he_init, needs this parameter to make a distribution
            neuron.initializeWeights(len(model[layer - 1]))




    #training starts here
    for minibatch in range(0, len(minibatches)):
        
        cost[minibatch] = [] #holds the array of costs associated with this minibatch

        for value in range(0, len(minibatches[minibatch])):
            
            
            #this represents one datapoint in the minibatch
            input_parameter = minibatch[minibatches][value]
            
            # probability_distribution holds the confidence of each output neuron
            probability_distribution = propagate_forward(input_parameter, model)
            
            # int for bias, vec for weights
            cost[minibatch][value] = []
            
            # cost_function compares probalilty vector with expected values and returns
            cost[minibatch][value] = cost_function(input_parameter, probability_distribution)



            


        
