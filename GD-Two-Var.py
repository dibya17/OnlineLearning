from numpy import *

batch_size_perc=0.8
# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(m, points):
    totalError = 0
    totalError= sum((points[:,3] - dot(m , transpose(points[:,:3]))) ** 2)
    #print totalError.shape
    return totalError / float(len(points))

def step_gradient(current_w, points, learningRate):
    init_gradient = 0
    N = float(len(points))
    #Vectorize the computation ,shape(X)=n * 4,shape(current_w)=1*4
    #cost function = (Y-MX)^2
    #optimize for M
    #first derivative = -2*X(Y-M*X)

    init_gradient += -(2/N) *dot((points[:,3] - dot(current_w, transpose(points[:,:3]))),points[:,:3])

    #compute the the updated weights by subtracting the learning rate
    new_m = current_w - (learningRate * init_gradient)
    return new_m

def gradient_descent_runner(points, starting_weights, learning_rate, num_iterations):
    m = starting_weights
    for i in range(num_iterations):
        prev_m=m
        m = step_gradient(prev_m, array(points), learning_rate)
        #if the error rate between two consecutive iteration reduces to less than threshold, exit
        if(abs(compute_error_for_line_given_points(m,points)-
                compute_error_for_line_given_points(prev_m,points))<1e-7):
            print "terminating in "+str(i)+" iterations"
            break
    return [m]

def stochastic_gradient_descent_runner(points, starting_weights, learning_rate, num_iterations):
    sampleSize=len(points)
    m = starting_weights
    for i in range(num_iterations):
        batch_size=min(int(batch_size_perc*sampleSize),sampleSize)
        random.shuffle(points)
        prev_m=m
        m = step_gradient(prev_m, array(points[:batch_size]),\
                          learning_rate)
        #if the error rate between two consecutive iteration reduces to less than threshold, exit
        if(abs(compute_error_for_line_given_points(m,points[:batch_size])-
                compute_error_for_line_given_points(prev_m,points[:batch_size]))<1e-7):
            print "terminating in "+str(i)+" iterations"
            break
    return [m]


def run():
    #define hyperparameters learning rate & iterations
    learning_rate = 0.0001
    num_iterations = 10000

    #read data from kagle train set for housing
    points = genfromtxt("train.csv", delimiter=",",skip_header=1,usecols=(4,46,80),dtype=float32)
    N=len(points)
    print type(points)
    #set the initial weights from normal distribution curve
    initial_weights = random.standard_normal(size=(1,3))

    #normalize by centering and scaling of data
    points[:,0] = (points[:,0]-( points[:,0]).mean())/abs(( points[:,0]).max()-( points[:,0]).min())
    points[:,1] = (points[:,1]-( points[:,1]).mean())/abs(( points[:,1]).max()-( points[:,1]).min())
    points[:,2] = (points[:,2]-( points[:,2]).mean())/abs(( points[:,2]).max()-( points[:,2]).min())
    pointsM=points

    #create a row of 1 corresponding to constant C in y=mx+c i.e. Y=M*X
    pointsM = insert(pointsM, 0, ones(len(points)),axis=1)

    #shuffle the dataset
    random.shuffle(pointsM)

    #devide the dataset to training and testing dataset
    trainPoints=pointsM[:int(N*0.8),:]
    testPoints=pointsM[int(N*0.8):,:]

    print "Starting gradient descent at weights = {0},error = {1}".format(initial_weights,\
    compute_error_for_line_given_points(initial_weights, trainPoints))
    print "Running..."
    m = stochastic_gradient_descent_runner(trainPoints, initial_weights, learning_rate, num_iterations)
    print "After {0} iterations weights = {1}, error = {2}".format(num_iterations, m\
    , compute_error_for_line_given_points(m, trainPoints))

    #test error computation
    print "Error before gradient descent = {0},error after gradient descent = {1}".\
        format(compute_error_for_line_given_points(initial_weights, testPoints),\
    compute_error_for_line_given_points(m, testPoints))

if __name__ == '__main__':
    run()
