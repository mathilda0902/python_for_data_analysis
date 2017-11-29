from pandas import Series, DataFrame
import pandas as pd
import numpy as np
'''Series'''
obj = Series([4,7,-5,3])
obj.values
'''Out[3]: array([ 4,  7, -5,  3])'''
obj.index
'''RangeIndex(start=0, stop=4, step=1)'''

obj2 = Series([4,7,-5,3], index=['d', 'b', 'c', 'a'])

'''In [7]: obj2.index
Out[7]: Index([u'd', u'b', u'c', u'a'], dtype='object')'''

obj2['d'] = 6
obj2[['c', 'a', 'd']]
'''Out[8]:
c   -5
a    3
d    6
dtype: int64'''

'''numpy operations, filtering with boolean array and scalar multiplication
will preserve the index-value link:'''
obj2[obj2 > 0]
'''Out[10]:
d    6
b    7
a    3
dtype: int64'''
np.exp(obj2)

'''Series as a fixed-length, ordered dict, mapping index values to data values'''
'e' in obj2
'''False'''

'''passing a dict into Series; index will be keys, in order'''
my_dict = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(my_dict)

states = ['California', 'Ohio', 'Texas', 'Oregon'] #index can be re-assigned
obj4 = Series(my_dict, index=states)

'''In [18]: obj4
Out[18]:
California        NaN
Ohio          35000.0
Texas         71000.0
Oregon        16000.0
Utah           5000.0
dtype: float64'''

'''detect missing data:'''
pd.isnull(df):      True if there is missing data
pd.notnull(df):     True if data not missing
'''for Series, these are instance methods:'''
my_series.isnull(), my_series.notnull()

'''Series and its index both have a 'name' feature:'''
obj4.name = 'population'
obj4.index.name = 'state'

'''DataFrame: creating method 1'''
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data, columns=['year', 'state', 'pop']) #specify col sequence
'''
Out[25]:
   year   state  pop
0  2000    Ohio  1.5
1  2001    Ohio  1.7
2  2002    Ohio  3.6
3  2001  Nevada  2.4
4  2002  Nevada  2.9'''

df2 = pd.DataFrame(data, columns=['year', 'state', 'pop'],
                index=['one', 'two', 'three', 'four', 'five']) #indexing
'''Out[27]:
       year   state  pop
one    2000    Ohio  1.5
two    2001    Ohio  1.7
three  2002    Ohio  3.6
four   2001  Nevada  2.4
five   2002  Nevada  2.9'''

df2.columns
'''Out[28]: Index([u'year', u'state', u'pop'], dtype='object')'''

'''retrive columns by two ways: by attribute, by dict-like notation:'''
df2['state']
df2.state

'''Assigning a column:
If assigning a list or an array: length must match
If assigning a Series: auto-fill in missing values, conformed to the DF's index
'''
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
df2['debt'] = val
df2

'''       year   state  pop  debt
one    2000    Ohio  1.5   NaN
two    2001    Ohio  1.7  -1.2
three  2002    Ohio  3.6   NaN
four   2001  Nevada  2.4  -1.5
five   2002  Nevada  2.9  -1.7'''

'''Creating a boolean new column:'''
df2['eastern'] = df2.state=='Ohio'
df2

'''       year   state  pop  debt  eastern
one    2000    Ohio  1.5   NaN     True
two    2001    Ohio  1.7  -1.2     True
three  2002    Ohio  3.6   NaN     True
four   2001  Nevada  2.4  -1.5    False
five   2002  Nevada  2.9  -1.7    False'''

del df2['eastern']
df2.columns
'''Out[42]: Index([u'year', u'state', u'pop', u'debt'], dtype='object')'''


'''DataFrame: creating method 2'''
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
        'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

df3 = pd.DataFrame(pop)
df3.T

'''if index is specified, will be imposed: losing data and NaN added.'''
pd.DataFrame(pop, index=[2001, 2002, 2003])
'''      Nevada  Ohio
2001     2.4   1.7
2002     2.9   3.6
2003     NaN   NaN'''

'''DF.values: returns values as 2D array:'''
df3.values
'''Out[53]:
array([[ nan,  1.5],
       [ 2.4,  1.7],
       [ 2.9,  3.6]])'''

df2.values
'''Out[54]:
array([[2000, 'Ohio', 1.5, nan],
       [2001, 'Ohio', 1.7, -1.2],
       [2002, 'Ohio', 3.6, nan],
       [2001, 'Nevada', 2.4, -1.5],
       [2002, 'Nevada', 2.9, -1.7]], dtype=object)''' #dtype fit all cols

'''Index objects are immutable: cannot be modified by assigning.'''
obj = pd.Series(range(3), index=['a', 'b', 'c'])
'Wrong!' index[1] = 'd'
'Correct!' index2 = pd.Index(np.arange(3))
            obj2 = pd.Series(range(3), index=index2)
obj2.index is index2
'''True'''

'''Index functions as a fixed-size set: set logic
append:         concatenate with other Index objects, producing a new Index
diff:           set difference as Index
intersection:   set intersection as Index
union:          set union as Index
isin:           boolean array
delete:         new Index with element at index i deleted
drop:           new Index by deleting passed values
insert:         new Index by inserting element at index i
is_monotonic:   True, if element is >= previous element
is_unique:      True, if Index has no duplicates
unique:         returns the array of unique values in the Index'''























1
