import  numpy as np
from matplotlib import pyplot as plt

print('---------1、创建标量、行向量、列向量、矩阵；-----------')
# 标量
x = 1
print('x:',x)
# 行向量
col = np.array([[1,2,3]])
print('col:',col)
print('col shape:',np.shape(col))
# 列向量
row = np.array([[1],[2],[3]])
print('row:',row)
print('row shape:',np.shape(row))
# 矩阵
matrix1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('matrix:',matrix1)
print('matrix shape:',np.shape(matrix1))

print('-------2、基本函数与操作；----------')
print('-------（1）创建一个向量（数组），求出向量的长度、转置、各元素之和、最大值、最小值；----------')
arr = np.array([[1,2,3,4,5]])
length = len(arr)
transposition = arr.T
sum = np.sum(arr)
max = np.max(arr)
min = np.min(arr)
print('length:',length)
print('transposition:',transposition)
print('sum:',sum)
print('max:',max)
print('min:',min)

print('-------（2）创建一个二维矩阵，取出矩阵的某一个元素、某一行、某一列、某一子矩阵；----------')
matrix2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
matrix2_num1 = matrix2[0,0]
matrix2_col1 = matrix2[0,:]
matrix2_row1 = matrix2[:,0]
print('matrix2_num1:',matrix2_num1)
print('matrix2_col1:',matrix2_col1)
print('matrix2_row1:',matrix2_row1)

print('-------（3）创建两个二维方阵，计算两个矩阵的和、积、点积，计算两个矩阵横向连接和纵向连接后的新矩阵；----------')
matrix3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
matrix4 = np.array([[7,8,9],[4,5,6],[1,2,3]])
sum = matrix3 + matrix4
multiply = matrix3 * matrix4
dot_multiply = np.dot(matrix3,matrix4)
connect_vertical = np.concatenate((matrix3,matrix4))
connect_level = np.concatenate((matrix3,matrix4),axis=1)
print('sum:',sum)
print('multiply:',multiply)
print('doc_multiply:',dot_multiply)
print('connect_vertical:',connect_vertical)
print('connect_level:',connect_level)

print('-------（4）创建一个向量，找出向量中所有大于 0 的元素的下标。----------')
arr2 = np.array([-1,2,-9,5,-4,-5,5,6,7,-3])
index = np.arange(0,10)
print('index:',index[arr2 > 0])

print('-------3、自定义函数（可在网上搜索关于def的帮助文档）----------')
print('-------（1）自定义一个函数，输入为两个实数，输出为此两个数的和、差、积、商；----------')
def calculate(num1,num2):
    sum = num1 + num2
    diff = num1 - num2
    multiply = num1 * num2
    divide = num1 / num2
    return 'sum:',sum,'diff:',diff,'multiply:',multiply,'divide:',divide
print(calculate(6,3))

print('-------（2）自定义一个函数，输入为一个区间的左边界、右边界和步长，以该步长在此区 间内绘制出函数 f(x)=x*sin(x)的图像；----------')
def paint(left,right,len):
    x = np.arange(left, right,len)
    y = x * np.sin(x)
    plt.plot(x,y)
    plt.show()
paint(-2 * np.pi,2 * np.pi,0.1)

print('-------（3）自定义一个函数，输入为一个向量，输出为向量中所有比前一个元素大的当前元素之和。----------')
def calculate_sum(arr):
    sum = 0
    for i in np.arange(1,len(arr)):
        if arr[i-1] < arr[i]:
            sum += arr[i]
    return sum
print('sum:',calculate_sum([1,2,3,4,5]))

