# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np

# Classifier class that implements multivariate linear regression using batch gradient descent
# algorithm works by training a vector of weights corresponding to each feature found in the environment
# the weight of a feature within the environment then can be used to calculate the most likely direction pacman moves.
class Classifier:
    def __init__(self):
        self.weights = []

    # function to convert number to its respective verbose direction
    # converts integer value to its respective direction as a string
    def convertIntegerToDirection(self, number):
        if number == 0:
            return "North"
        elif number == 1:
            return "East"
        elif number == 2:
            return "South"
        elif number == 3:
            return "West"
        elif number<0 or number>3:
            return "Random Legal Direction"
     
    # checks if the direction predicted is a legal move  
    # returns relevant string to represent legality 
    def legalMoveCheck(self, direction, legal):
        
        for move in legal:
            if move == direction:
                return "Legal => Move Made"
        return "Illegal => Random Legal Move Made"
                
    
    def reset(self):
        pass
    
    # implements a batch gradient descent multivariate linear regression classification,
    # defines constants for learning rate, tolerance (to stop iterating after convergence) and regularisation (lambda in given formulas)
    # the function takes an input vector and an expected outcome vector
    # will assign each feature in the vector a weight which gets updated till 
    # weight recalculations are within 1 percent of each other.
    def fit(self, data, target, learning_rate=0.01, tolerance=0.0001, regularisation = 0.01):
        nd_data = np.array(data)
        feature_size = nd_data.shape[1]
        new_weights = np.zeros(feature_size)
        new_weights[...] = 0.1 #initialise initial feature weights  to 0.1
        old_loss = float('inf') #initialise previous loss to an infinite float
        epoch = 0
        
        # iterate repeatedly to update the weights 
        while True:
            print(f' Training {epoch} epochs',  end='\r')  
            predictions = np.dot(data, new_weights) # dot product of feature vector and weight vector to find predictions
            errors = target-predictions
            loss = np.sum(errors ** 2) + (regularisation * np.sum(new_weights ** 2)) # regularisation term applied to loss

            if epoch > 0 and abs((old_loss - loss) / old_loss) <= tolerance: # check if losses are now within 0.01% of each other
                print(f' Converged at {epoch} epochs')                      #break out of training loop if it is the case
                break
            
            old_loss = loss
            
            new_weights += learning_rate * np.dot(nd_data.T, errors) # calculate and add new weights to new_weights
            epoch+=1
       
        # mutate the class field for weights
        self.weights = new_weights
        
    # function the takes an input vector (in this case the vector representing
    # the current state of the pacman environment) and returns a prediction based off of the
    # dot product of the input data and the weights of each feature in the vector trained by the fit function.
    # also prints f string containing rounded prediction, legal moves, the direction represented by the prediction integer and if that move was legal.
    def predict(self, data, legal=None):    
        prediction = np.dot(data, self.weights)
        
        # check if value calculated is NaN to avoid errors before rounding
        if np.isnan(prediction):
            prediction=-1
            
        # round the predicted value so it is valid for conversion to direction.
        rounded_prediction = round(prediction)
        verbose_direction = self.convertIntegerToDirection(rounded_prediction)
        verbose_legality = self.legalMoveCheck(verbose_direction, legal)

        print(f"Rounded Prediction: {rounded_prediction}, Legal Moves: {legal}, Direction: {verbose_direction} ({verbose_legality}) ")
        
        return rounded_prediction
