import numpy as np
import xlrd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show, subplot, hist, ylim
from scipy.linalg import svd

doc = xlrd.open_workbook('./resources/hungarian8mkclean.xls').sheet_by_index(0)

# Extract attribute names (1st row, column 0 to 7)
attributeNames = doc.row_values(0, 0, 8)



print(attributeNames)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(7, 1, 295)
#print(classLabels)
#print(len(doc.col_values(7)))
classNames = sorted(set(classLabels))

classDict = dict(zip(classNames, range(0,2)))


print(classDict)


# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((294, 8))
for i, col_id in enumerate(range(0, 8)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 295))



# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

print(N)
print(M)
print(C)




# Data attributes to be plotted
i = 2
j = 3

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X) #Try to uncomment this line
#plot(X[:, i], X[:, j], 'o')


# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
title('Heart Disease data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o')

legend(classNames)

xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()

figure(figsize=(12,12))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i], color=(0.2, 0.4, 0.4))
    xlabel(attributeNames[i])
    ylim(0,N/2)
    
show()

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
V = V.T
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 


# Project the centered data onto principal component space
Z = Y @ V

figure()
plot(range(1,len(rho)+1),rho,'o-')
title('Variance explained by principal components')
xlabel('Principal component')
ylabel('Variance explained')
show()


# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title(': PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o')
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()