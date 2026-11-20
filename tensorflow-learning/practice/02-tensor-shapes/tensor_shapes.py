"""把 shape 摸清楚：索引、切片、reshape、矩阵乘、广播。

重点不是背 API，而是每做一步都知道：
1. 现在 shape 是什么；
2. 为什么能算 / 为什么报错；
3. 结果是几行几列。

Run:
    conda activate tf-esp
    python tensor_shapes.py
"""

import tensorflow as tf

# 2 行 3 列的整数矩阵，shape 是 (2, 3)。
# 第一个维度是行，第二个维度是列。
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])

# 长度为 3 的浮点向量，shape 是 (3,)。
# 注意 (3,) 是一维；它不是 (3,1)，也不是 (1,3)。
vector = tf.constant([1.0, 2.0, 3.0])

print("matrix:")
print(matrix)
print("shape:", matrix.shape)

# matrix[0] 取第 0 行，结果 shape 是 (3,)。
# Python 下标从 0 开始，所以 0 是第一行。
print("\nrow 0:")
print(matrix[0])

# matrix[:, 1] 里：
#   :   表示“所有行”
#   1   表示第 1 列
# 所以它取的是“所有行的第 1 列”，结果 shape 是 (2,)。
# 这里最容易看错：它不是第 1 行。
print("\ncolumn 1:")
print(matrix[:, 1])

# 两个下标：第 0 行、第 1 列，得到一个标量。
# 标量的 shape 是 ()，不是 (1,)。
print("\nelement (0, 1):")
print(matrix[0, 1])

# 切片：0:2 表示从下标 0 开始，到下标 2 之前（不包含 2）。
# 所以是前两行、前两列，结果 shape 是 (2, 2)。
print("\nslice rows 0:2, cols 0:2:")
print(matrix[0:2, 0:2])

# tf.reshape 改形状，不改数据顺序。
# 原来的 (2,3) 有 6 个元素，改成 (3,2) 也必须是 6 个元素。
# 数据按行优先重新排：
#   [1,2,3,4,5,6] -> [[1,2],[3,4],[5,6]]
print("\nreshape to (3, 2):")
print(tf.reshape(matrix, (3, 2)))

print("\nvector:", vector, "shape:", vector.shape)

# reshape 里的 -1 表示“这一维你自己算”。
# 这里元素总数是 3，所以 -1 会被算成 3。
# column 是 (3,1)：3 行 1 列。
column = tf.reshape(vector, (-1, 1))

# row 是 (1,3)：1 行 3 列。
row = tf.reshape(vector, (1, -1))

print("\nas column, shape:", column.shape)
print(column)
print("\nas row, shape:", row.shape)
print(row)

# 广播加法：
# column 是 (3,1)，vector 是 (3,)。
# 从最右边对齐时，(3,) 会被当成 (1,3)。
# 逐维比较：3 和 1 -> 3；1 和 3 -> 3，所以输出是 (3,3)。
# 数据上相当于“每一行加同一个向量”，得到 3x3 的矩阵。
print("\ncolumn + vector (broadcasting):")
print(column + vector)

# a 和 b 都是 2x2 浮点矩阵。
a = tf.constant([[1., 2.],
                 [3., 4.]])
b = tf.constant([[5., 6.],
                 [7., 8.]])

# @ 是矩阵乘：
# (2,2) @ (2,2) -> (2,2)
# 要求左边列数等于右边行数。
# 结果是“行乘列再求和”，不是逐元素乘。
print("\na @ b:")
print(a @ b)

# * 是逐元素乘：
# 对应位置相乘，shape 必须能广播或完全相同。
print("\na * b:")
print(a * b)

# transpose 交换行和列：
# (2,3) -> (3,2)
print("\ntranspose of matrix:")
print(tf.transpose(matrix))

# bias 的 shape 是 (3,)。
# matrix 是 (2,3)，最后一维 3 和 bias 的 3 对上，
# 所以每一行都加 [10,20,30]。
bias = tf.constant([10, 20, 30])       # (3,)

# row_bias 的 shape 是 (2,1)。
# 第一维 2 和 matrix 的行数 2 对上，
# 所以第 0 行加 100，第 1 行加 200。
row_bias = tf.constant([[100], [200]]) # (2, 1)

print("\nmatrix + bias, (2,3) + (3,):")
print(matrix + bias)

print("\nmatrix + row_bias, (2,3) + (2,1):")
print(matrix + row_bias)

# (3,1) + (1,3)：
# 第 0 维：3 和 1 -> 3
# 第 1 维：1 和 3 -> 3
# 结果是 (3,3)，相当于“列 + 行”的外和。
print("\ncolumn + row, (3,1) + (1,3):")
print(column + row)
