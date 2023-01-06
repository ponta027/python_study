########################
#
########################
import ax

version = ax.__version__
print(version)


"""
f(x) = x_{1}^{2}+x_{2}^{2}

x [-10,10] i=1,2

range,bounds,[-10,10]
fixed,value,10
choice,values,[1,10,100]
"""

parameters = [
        {'name':'x1','type':'range','bounds':[-10.,10.]},
        {'name':'x2','type':'range','bounds':[-10.,10.]}
        ]

def evaluation_function( parameters ):
    x1 = parameters.get('x1')
    x2 = parameters.get('x2')
    f = x1**2 + x2**2
    return f


results = ax.optimize(parameters,evaluation_function,minimize = True, random_seed = 0)
(best_parameters , values, experiment , model ) = results
print(best_parameters)
print(value)


