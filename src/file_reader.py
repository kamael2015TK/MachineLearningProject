##
# Author Duran KöseS147153
#
import xlrd
import numpy as np

doc = xlrd.open_workbook('./resources/hungarian8mkclean.xls').sheet_by_index(0)
attributeNames = doc.row_values(0, 0, 8)
# Preallocate memory, then extract excel data to matrix X
data = np.empty((294, 8))
for i, col_id in enumerate(range(0, 8)):
    data[:, i] = np.asarray(doc.col_values(col_id, 1, 295))

# By running this script you will get 
# data => multi array [8][294] 
# attributeNames => Feature names