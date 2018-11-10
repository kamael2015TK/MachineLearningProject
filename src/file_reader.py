import xlrd
import numpy as np

doc = xlrd.open_workbook('./resources/hungarian8mkclean.xls').sheet_by_index(0)
attributeNames = doc.row_values(0, 0, 8)
classLabels = doc.col_values(7, 1, 295)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(0,2)))
y = np.asarray([classDict[value] for value in classLabels])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((294, 8))
for i, col_id in enumerate(range(0, 8)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 295))