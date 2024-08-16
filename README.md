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
