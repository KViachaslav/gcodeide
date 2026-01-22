import numpy as np
A = 39
B = 43.83
ang = 45.3
d = 12-0.424

print(d * (1-np.cos(np.deg2rad(ang)))/np.sin(np.deg2rad(ang)))
print(B-A)