# nn-from-scratch

this is the readme for my 2nd ai project. I will be making a NN from scratch in python, using concepts I learned in 3Blue1Brown's series on NNs. 

"from scratch" means I'll not be using any python AI libraries like PyTorch or TensorFlow. Instead, I'll just be using numpy to help with the mathematics behind it.

When I get stuck, I will be refernecing https://www.youtube.com/watch?v=w8yWXqWQYmU I think it is a great resource for this type of learning.

Notes on learning:
    - the activation function chosen is the ReLU function, which given an input x returns the max between 0 and x. max(0, x)
    - the process for initializing weights + biases is called "He initialization" which overcomes the "ReLU dying" problem
        - this happens when an activation function consistiently outputs 0 (because the weight attached is too low) 
        - He initialization starts weights off at higher values than normal to avoid this
        - He initialization has a mean of 0 and a variance of 2/n, where n is the number of neurons feeding into this one
        - upon further learning, while he initialization does improve relu death, it is not ideal for preventing it. instead, an asymmetric distribution (he is normal aka symmetric) which favors positive numbers over negatives would be better to avoid relu death. here's an article on the subject: https://towardsdatascience.com/the-dying-relu-problem-clearly-explained-42d0c54e0d24
        ways this article suggests to prevent relu death:
        - use a lower learning rate. In backpropagation, a new weight value is the difference between the old weight and the learning rate multiplied by the error w.r.t. the old weight.
        - avoid negative biases
        - use an asymmetric distribution for initializing weights, one that favors positive numbers
        - use a variation of relu that doesn't output 0 for all negative numbers, but continues some path
