import math
from typing import List

import numpy as np
import numpy.random as nprand
class SimpleNetwork:
	@classmethod
	def of(cls,n_input: int,n_hidden: int,n_output: int):
		def uniform(n_in,n_out):
			epsilon=math.sqrt(6)/math.sqrt(n_in+n_out)
			return nprand.uniform(-epsilon,+epsilon,size=(n_in,n_out))
		return cls(uniform(n_input,n_hidden),uniform(n_hidden,n_output))

	def __init__(self,input_to_hidden_weights:np.ndarray,hidden_to_output_weights: np.ndarray):
		self.input_to_hidden_weights=input_to_hidden_weights
		self.hidden_to_output_weights=hidden_to_output_weights

	def sigmoid(self,x):
		return (1/(1+np.exp(-x)))

	def predict(self, input_matrix: np.ndarray) -> np.ndarray:
		self.hidden_input=np.dot(input_matrix, self.input_to_hidden_weights)          # Dot product  of input_matrix and input_to_hidden_weights
		self.hidden_output=self.sigmoid(self.hidden_input)							  # hidden_output =  Sigmoid of the dot product
		self.output_input=np.dot(self.hidden_output, self.hidden_to_output_weights)	  # Dot product  of ihidden_output and hidden_to_output_weights
		self.output_output=self.sigmoid(self.output_input)                			  # output_output = sigmoid of the dot product obtained
		return self.output_output

	def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
		predict_array=self.predict(input_matrix)								   	  # predict_array is the values predicted from the input matrix
		predict_zero_one_array=np.zeros(predict_array.shape)	                      # predicted_zero_one_array is initialized to all zeros
		a=0																			  # a and b are counter variables to assign 0s and 1s accordingly in predicted_zero_one_array
		for i in predict_array:
			b=0
			for j in i:
				if predict_array[a][b] >= 0.5:
					predict_zero_one_array[a][b] = 1
				else:
					predict_zero_one_array[a][b] = 0
				b+=1
			a+=1		
		return predict_zero_one_array


	def gradients(self,input_matrix: np.ndarray,output_matrix: np.ndarray) -> List[np.ndarray]:
		y_predicted = self.predict(input_matrix)
		error_3 = y_predicted - output_matrix                                     # calculates the error
		a = self.hidden_output                                                    # a is the result of activation of the hidden layer
		hidden_output_gradients = (np.dot(a.transpose(), error_3)) / len(input_matrix)    # (dot(Atranspose,error_3)/ total number of inputs)

		temp = (np.dot(error_3, self.hidden_to_output_weights.transpose())) * (a*(1-a))   #temp=(dot(error_3,InputMatrixTranspose) * a(1-a))
		input_hidden_gradients = np.dot(input_matrix.transpose(),temp) / len(input_matrix)  #(dot(inputMatrixTranspose, temp) / total number of inputs)

		final_gradients = []      
		final_gradients.append(input_hidden_gradients)                           
		final_gradients.append(hidden_output_gradients)
		return final_gradients

	def train(self,input_matrix: np.ndarray,output_matrix: np.ndarray,iterations: int = 10,learning_rate: float = 0.1) -> None:
		
		
		for i in range(0,iterations):  #for each interation, calculate the gradients and update the input_to_hidden_weights and hidden_to_output_weights 
			[input_hidden_gradients,hidden_output_gradients]=self.gradients(input_matrix,output_matrix)
			self.input_to_hidden_weights=self.input_to_hidden_weights - (learning_rate * input_hidden_gradients)
			self.hidden_to_output_weights=self.hidden_to_output_weights - (learning_rate * hidden_output_gradients)
			





		











