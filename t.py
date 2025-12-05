import numpy as np
def get_radius(a,b,x,y):
    if y == 0:
        return b
    if x == 0:
        return a
    else:
        ans = a*b/np.sqrt((b*np.cos(np.arctan(x/y)))**2 + (a*np.sin(np.arctan(x/y)))**2) 
        return ans + (a-ans) * 0.5


print(get_radius(10,5,0,3))