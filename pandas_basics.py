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
unique:         returns the array of unique values in the Index
'''

'''reindexing:'''
obj = pd.Series([1,2,3], index=['d', 'c', 'a'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
'''
In [5]: obj2
Out[5]:
a    3.0
b    NaN
c    2.0
d    1.0
e    NaN
dtype: float64'''

obj3 = obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
'''a    3
b    0
c    2
d    1
e    0
dtype: int64'''

obj4 = pd.Series(['blue', 'purple', 'yellow'], index=[0,2,4])
obj5 = obj4.reindex(range(6), method='ffill')
'''0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow
dtype: object'''

'''reindex methods:
ffill / pad: fill or carry values forward
bfill / backfill: fill or carry values backward'''

'''reindex DataFrame:'''
frame = pd.DataFrame(np.arange(9).reshape((3,3)), index=['a', 'c', 'd'],
                    columns=['Ohio', 'Texas', 'California'])

'''In [16]: frame.reindex(['a', 'b','c', 'd'])
Out[16]:
   Ohio  Texas  California
a   0.0    1.0         2.0
b   NaN    NaN         NaN
c   3.0    4.0         5.0
d   6.0    7.0         8.0'''
'''In [17]: frame.reindex(columns=['Texas', 'Utah', 'California'])
Out[17]:
   Texas  Utah  California
a      1   NaN           2
c      4   NaN           5
d      7   NaN           8'''

'''using .ix to reindex:'''
frame.ix[['a','b','c','d'], ['Texas', 'Utah', 'California']]
'''Out[28]:
   Texas  Utah  California
a    1.0   NaN         2.0
b    NaN   NaN         NaN
c    4.0   NaN         5.0
d    7.0   NaN         8.0'''

'''reindex function arguments:
index:      sequence to use as index
method:     interpolation methods
fill_value: to fill in for missing data, when introduced by reindexing
limit:      maximum size gap to fill, when forward-, or backfilling
copy:       True, then copy underlying data even indexes are equivalent;
            False, do not copy data when indexes are equivalent.'''

'''drop entries from an axis:'''
data = pd.DataFrame(np.arange(16).reshape((4,4)),
            index=['Ohio', 'Colorado', 'Utah', 'New York'],
            columns=['one', 'two', 'three', 'four'])

'''In [31]: data
Out[31]:
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15'''

data.drop(['Colorado', 'Ohio'])
'''Out[32]:
          one  two  three  four
Utah        8    9     10    11
New York   12   13     14    15'''

data.drop('two', axis=1)
'''Out[33]:
          one  three  four
Ohio        0      2     3
Colorado    4      6     7
Utah        8     10    11
New York   12     14    15'''


data.drop(['two', 'four'], axis=1)
'''Out[34]:
          one  three
Ohio        0      2
Colorado    4      6
Utah        8     10
New York   12     14'''

'''slicing includes the endpoint; can be re-assigned value:'''
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj['b':'c'] = 5
obj
'''Out[39]:
a    0.0
b    5.0
c    5.0
d    3.0
dtype: float64'''

'''ix way of indexing a subset of the rows and columns.'''
data.ix['Colorado', ['two', 'three']]

'''arithmetic methods with fill values:'''
df1 = pd.DataFrame(np.arange(12.).reshape((3,4)), columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4,5)), columns=list('abcde'))

'''In [44]: df1+df2
Out[44]:
      a     b     c     d   e
0   0.0   2.0   4.0   6.0 NaN
1   9.0  11.0  13.0  15.0 NaN
2  18.0  20.0  22.0  24.0 NaN
3   NaN   NaN   NaN   NaN NaN'''

df1.add(df2, fill_value=0)
'''Out[45]:
      a     b     c     d     e
0   0.0   2.0   4.0   6.0   4.0
1   9.0  11.0  13.0  15.0   9.0
2  18.0  20.0  22.0  24.0  14.0
3  15.0  16.0  17.0  18.0  19.0'''

df1.reindex(columns=df2.columns, fill_value=0)
'''Out[47]:
     a    b     c     d  e
0  0.0  1.0   2.0   3.0  0
1  4.0  5.0   6.0   7.0  0
2  8.0  9.0  10.0  11.0  0'''

'''flexible arithemetic methods:
add: +
sub: -
div: /
mul: *'''

'''Series broadcasting to DataFrame, by rows or by columns:'''
frame = pd.DataFrame(np.arange(12.).reshape((4,3)), columns=list('bde'),
                    index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
'''In [52]: frame
Out[52]:
          b     d     e
Utah    0.0   1.0   2.0
Ohio    3.0   4.0   5.0
Texas   6.0   7.0   8.0
Oregon  9.0  10.0  11.0

In [54]: series
Out[54]:
b    0.0
d    1.0
e    2.0
Name: Utah, dtype: float64'''

frame - series #broadcasting down the rows
'''Out[55]:
          b    d    e
Utah    0.0  0.0  0.0
Ohio    3.0  3.0  3.0
Texas   6.0  6.0  6.0
Oregon  9.0  9.0  9.0'''

series2 = pd.Series(range(3), index=list('bdf'))
frame + series2 #broadcasting: union of columns, but non-NAN only for intersecting columns
'''Out[56]:
          b     d   e   f
Utah    0.0   2.0 NaN NaN
Ohio    3.0   5.0 NaN NaN
Texas   6.0   8.0 NaN NaN
Oregon  9.0  11.0 NaN NaN'''

series3 = frame['d']
'''Out[58]:
Utah       1.0
Ohio       4.0
Texas      7.0
Oregon    10.0
Name: d, dtype: float64'''

frame.sub(series3, axis=0) # braodcasting: axis=0, series onto frame by columns
'''Out[62]:
          b    d    e
Utah   -1.0  0.0  1.0
Ohio   -1.0  0.0  1.0
Texas  -1.0  0.0  1.0
Oregon -1.0  0.0  1.0'''

'''mapping functions by columns (axis=0), rows (axis=1)'''
frame = pd.DataFrame(np.random.randn(4,3), columns=list('bde'),
                    index=['Utah', 'Ohio', 'Texas', 'Oregon'])
f = lambda x: x.max() - x.min()
frame.apply(f)
'''Out[68]:
b    1.646147
d    2.385127
e    0.287692
dtype: float64'''

frame.apply(f, axis=1)
'''Out[69]:
Utah      0.493539
Ohio      0.894958
Texas     1.420720
Oregon    2.493630
dtype: float64'''

def f(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])

frame.apply(f)
'''Out[71]:
            b         d         e
min -0.763506 -1.610989 -0.926531
max  0.882641  0.774138 -0.638839'''

'''element-wise broadcasting formatting function:'''
format = lambda x: '%.2f' % x
frame.applymap(format)
'''Out[73]:
            b      d      e
Utah    -0.44  -0.93  -0.77
Ohio    -0.76   0.13  -0.64
Texas    0.25   0.77  -0.65
Oregon   0.88  -1.61  -0.93'''

'''for Series, there is .map() to broadcast elementwise:'''
frame['e'].map(format)
'''Out[77]:
Utah      -0.77
Ohio      -0.64
Texas     -0.65
Oregon    -0.93
Name: e, dtype: object'''

'''sorting a DataFrame:
df.sort_index(axis=1, ascending=True)
sorting a Series:
series.order()
'''

'''options for reduction methods:
axis: 0 for columns, 1 for rows
skipna: default is True, exclude missing values
level: reduce grouped by level if the axis is multiIndex'''

'''redution methods, accumulation methods, and summary:
sum
mean(axis=1, skipna=False)
argmin, argmax:     returns index locations where min/max are attained
idxmax, idxmin:     returns index values where the min/max are attained
cumsum: accumulations
describe(): summary statistics in one shot
quantile
mad:                mean absolute deviation from mean value
count:              number of non-NAN values
skew:               sample 3rd moment of values
kurt:               sample 4th moment of values
cummin, cummax:     cumultive mim/max values
cumprod:            cum product
diff:               1st arithmetic difference, time series
pct_change:         percent changes
'''

'''correlation and covariance:
corr(), cov():      returns corr/cov matrix
col1.corr(col2), col1.cov(col2):    corr/cov value between two colummns
df.corrwith(col), df.covwith(col):      pairwise corr/cov with column 'col'

























1
