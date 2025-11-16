import numpy as np
from scipy import signal 
from layer import Layer

class Convolutional(Layer):
    def __init__(self,input_shape,kernal_size,depth):
        input_depth,input_height,input_width=input_shape
        self.depth=depth
        self.input_Shape=input_shape
        self.input_dept = input_depth
        self.output_shape=(depth,input_height-kernal_size+1,input_width-kernal_size+1)
        self.kernals_shape=(depth,input_depth,kernal_size,kernal_size)
        self.kernals=np.random.randn(*self.kernals_shape)
        self.biases=np.random.randn(*self.output_shape)
        
    def forward(self,input):
        self.input=input
        self.output=np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i]+=signal.correlate2d(self.input[j],self.kernals[i,j],"valid")
            return self.output
        
    def backward(self,output_gradient,learning_rate):
        kernals_gradient=np.zeros(self.kernals_shape)
        input_gradient=np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernals_gradient[i,j]=signal.correlate2d(self.input[j],output_gradient[i],"valid")
                input_gradient[j]+=signal.convolve2d(output_gradient[i],self.kernals[i,j],"full")
                
        self.kernals -=learning_rate*kernals_gradient
        self.biases -=learning_rate*output_gradient
        return input_gradient