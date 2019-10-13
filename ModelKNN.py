
# coding: utf-8

# In[30]:


import numpy as np
import os
import cv2
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


folder = os.listdir('./dataset/')


# In[9]:


x = []
y = []
for f in folder:
    files = os.listdir('./dataset/'+f)
    for img in files:
        y.append(f)
        x.append(cv2.imread('./dataset/'+f+'/'+img,cv2.IMREAD_GRAYSCALE))
        
x = np.asarray(x)
y = np.asarray(y)


# In[14]:


x = x/255.0


# In[17]:


from skimage.feature import hog


# In[22]:


features = []
for img in x:
    features.append(hog(image=img,orientations=8,pixels_per_cell=(7,7),cells_per_block=(4,4)))

features = np.array(features,'float64')


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(features,y)


# In[27]:





# In[43]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

model_score = knn.score(x_test,y_test)


# In[44]:


model_score


# In[103]:





# In[98]:


import pickle


# In[100]:


filename = 'finalized_model.sav'
pickle.dump(knn, open(filename, 'wb'))


# In[106]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)


# In[90]:


x= np.load("nums.npy")
temp = np.asarray([np.average(i) for i in x])

x = x[temp>5]
x = x/255.0


# In[91]:


test_features = []
for img in x:
    test_features.append(hog(image=img,orientations=8,pixels_per_cell=(7,7),cells_per_block=(4,4)))
    
test_features = np.array(test_features,'float64')


# In[92]:


cls = knn.predict(test_features)


# In[93]:


sudoku = []
ptr = 0
for i in range(81):
    if(temp[i] == False):
        sudoku.append(0)
    else:
        sudoku.append(cls[ptr])
        ptr += 1

sudoku = np.asarray(sudoku,dtype='int8')
sudoku = np.reshape(sudoku,(9,9))


# In[94]:


sudoku


# In[95]:


def findNextCellToFill(grid, i, j):
        for x in range(i,9):
                for y in range(j,9):
                        if grid[x][y] == 0:
                                return x,y
        for x in range(0,9):
                for y in range(0,9):
                        if grid[x][y] == 0:
                                return x,y
        return -1,-1

def isValid(grid, i, j, e):
        rowOk = all([e != grid[i][x] for x in range(9)])
        if rowOk:
                columnOk = all([e != grid[x][j] for x in range(9)])
                if columnOk:
                        # finding the top left x,y co-ordinates of the section containing the i,j cell
                        secTopX, secTopY = 3 *(i//3), 3 *(j//3) #floored quotient should be used here. 
                        for x in range(secTopX, secTopX+3):
                                for y in range(secTopY, secTopY+3):
                                        if grid[x][y] == e:
                                                return False
                        return True
        return False

def solveSudoku(grid, i=0, j=0):
        i,j = findNextCellToFill(grid, i, j)
        if i == -1:
                return True
        for e in range(1,10):
                if isValid(grid,i,j,e):
                        grid[i][j] = e
                        if solveSudoku(grid, i, j):
                                return True
                        # Undo the current cell for backtracking
                        grid[i][j] = 0
        return False


# In[96]:


solveSudoku(sudoku)
sudoku


# In[110]:


data = open('finalized_model.sav','rb')
dat = data.read()


# In[117]:


dat = str(dat)


# In[118]:


file = open('model','w')
file.write(dat)
file.close()

