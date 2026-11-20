# TF 学习中的 Python 笔记

> 用法：在 TensorFlow/Keras 代码里每遇到一个新的 Python 概念，就在下面加一条。
> 路线：Keras 为主线，TensorFlow 只在导出、转换、量化时补；Python 也只学这些环节实际遇到的部分。
> 一条笔记只回答四个问题：**是什么、最小例子、在 TF/Keras 哪里出现、有什么坑**。
> 不要为了“学完 Python”停下来；够用就继续。

---

## 笔记模板

```markdown
## 概念名

- **是什么**：
- **最小例子**：
- **在 TF 里哪里见到**：
- **坑**：
- **掌握程度**：认识 / 会用 / 熟练
```

---

## 1. import / 模块

- **是什么**：导入别的代码文件或包。
- **最小例子**：

  ```python
  import tensorflow as tf
  import numpy as np
  ```

- **在 TF 里哪里见到**：几乎每个脚本开头。
- **坑**：`ModuleNotFoundError` 通常是环境不对，不是代码错。

## 2. 变量与基本类型

- **是什么**：给值起名字；常见 int/float/str/bool。
- **最小例子**：

  ```python
  x = 3
  y = 0.5
  name = "sine"
  ok = True
  ```

- **在 TF 里哪里见到**：`epochs = 300`、`batch_size = 64`。
- **坑**：Python 不需要声明类型；但 TensorFlow 的张量有自己的 `dtype`。

## 3. print 与 f-string

- **是什么**：打印；f-string 把变量塞进字符串。
- **最小例子**：

  ```python
  x = 3
  print("x =", x)
  print(f"x = {x}")
  ```

- **在 TF 里哪里见到**：打印 shape、loss、版本。
- **坑**：`print(x.shape)` 比 `print(x)` 更适合排查维度。

## 4. 缩进

- **是什么**：Python 用缩进表示代码块，不用 `{}`。
- **最小例子**：

  ```python
  if True:
      print("yes")
  ```

- **在 TF 里哪里见到**：训练循环、`with`、函数体。
- **坑**：空格和 Tab 混用会报 `IndentationError`。

## 5. 列表 list

- **是什么**：有序、可变的一组值。
- **最小例子**：

  ```python
  xs = [1, 2, 3]
  xs.append(4)
  print(xs[0])
  ```

- **在 TF 里哪里见到**：`callbacks=[...]`、`[w, b]`、层的列表。
- **坑**：列表不是 NumPy 数组，不能直接做矩阵运算。

## 6. 元组 tuple

- **是什么**：有序、不可变的一组值。
- **最小例子**：

  ```python
  shape = (2, 2)
  print(shape[0])
  ```

- **在 TF 里哪里见到**：`x.shape`、`Input((1,))`、`(x, y)`。
- **坑**：单元素元组要写 `(1,)`，不是 `(1)`。

## 7. 字典 dict

- **是什么**：键值对集合。
- **最小例子**：

  ```python
  d = {"name": "sine", "size": 3592}
  print(d["size"])
  ```

- **在 TF 里哪里见到**：`history.history["loss"]`、`input_detail["shape"]`。
- **坑**：键不存在会 `KeyError`；先打印 `d.keys()`。

## 8. 索引与切片

- **是什么**：取列表/数组的一部分。
- **最小例子**：

  ```python
  xs = [0, 1, 2, 3]
  print(xs[0])      # 第一个
  print(xs[1:3])    # 1,2
  ```

- **在 TF 里哪里见到**：`x[:, 1]`、`x[0]`、`x[..., None]`。
- **坑**：Tensor 的切片和 NumPy 类似，但要注意 shape 变化。

## 9. if / for

- **是什么**：条件与循环。
- **最小例子**：

  ```python
  for i in range(3):
      if i % 2 == 0:
          print(i)
  ```

- **在 TF 里哪里见到**：训练循环、遍历 batch、判断文件是否存在。
- **坑**：`range(3)` 是 0,1,2；不是 0,1,2,3。

## 10. 函数 def

- **是什么**：把一段代码封装起来，可传参、可返回。
- **最小例子**：

  ```python
  def add(a, b):
      return a + b

  print(add(1, 2))
  ```

- **在 TF 里哪里见到**：`representative_dataset()`、回调、`@tf.function` 包裹的函数。
- **坑**：函数体必须缩进；返回值用 `return`。

## 11. 类与对象

- **是什么**：把数据和操作封装在一起；`model` 就是对象。
- **最小例子**：

  ```python
  class Counter:
      def __init__(self):
          self.n = 0

      def add(self):
          self.n += 1

  c = Counter()
  c.add()
  print(c.n)
  ```

- **在 TF 里哪里见到**：`model = tf.keras.Sequential(...)`、`model.fit(...)`。
- **坑**：`__init__` 是构造函数；方法第一个参数通常是 `self`。

## 12. 关键字参数

- **是什么**：调用时写参数名，顺序可以变。
- **最小例子**：

  ```python
  def train(epochs=10, batch_size=64):
      print(epochs, batch_size)

  train(batch_size=32, epochs=5)
  ```

- **在 TF 里哪里见到**：`model.fit(..., epochs=300, batch_size=64)`。
- **坑**：拼错参数名会 `TypeError`。

## 13. with

- **是什么**：上下文管理器，进入/退出时自动做清理。
- **最小例子**：

  ```python
  with open("a.txt", "w") as f:
      f.write("hello")
  ```

- **在 TF 里哪里见到**：`with tf.GradientTape() as tape:`。
- **坑**：`with` 块结束后资源会自动释放；不要在块外还用已关闭的文件。

## 14. 装饰器 @

- **是什么**：不改函数代码，给函数套一层功能。
- **最小例子**：

  ```python
  def deco(fn):
      def wrapper():
          print("before")
          fn()
          print("after")
      return wrapper

  @deco
  def hello():
      print("hello")

  hello()
  ```

- **在 TF 里哪里见到**：`@tf.function`。
- **坑**：先会“用它”即可，不必一开始就会写装饰器。

## 15. 生成器 yield

- **是什么**：一次产出一个值，适合大数据/流式数据。
- **最小例子**：

  ```python
  def numbers():
      for i in range(3):
          yield i

  for n in numbers():
      print(n)
  ```

- **在 TF 里哪里见到**：`representative_dataset()` 里 `yield [x]`。
- **坑**：`yield` 不是 `return`；函数调用后返回生成器，不会立刻执行完。

## 16. 异常与报错

- **是什么**：程序出错时抛出的异常。
- **最小例子**：

  ```python
  try:
      x = 1 / 0
  except ZeroDivisionError:
      print("cannot divide by zero")
  ```

- **在 TF 里哪里见到**：`ModuleNotFoundError`、`ValueError: shape mismatch`。
- **坑**：先看 Traceback 最后一行；再看最上面一行定位自己的代码。

## 17. __name__ == "__main__"

- **是什么**：只有直接运行脚本时才执行的入口。
- **最小例子**：

  ```python
  def main():
      print("run")

  if __name__ == "__main__":
      main()
  ```

- **在 TF 里哪里见到**：训练脚本入口。
- **坑**：被别的文件 import 时，这段不会执行。

## 18. NumPy 数组与 shape

- **是什么**：高效的数值数组；TF 数据通常先走 NumPy。
- **最小例子**：

  ```python
  import numpy as np

  x = np.linspace(0, 1, 5).reshape(-1, 1)
  print(x.shape)   # (5, 1)
  ```

- **在 TF 里哪里见到**：训练数据、`x.reshape(1, 96, 96, 1)`。
- **坑**：`(N,)` 和 `(N,1)` 不是一回事；TF 对 shape 很敏感。

---

## 快速索引

| 概念 | 阶段 | 优先级 |
|---|---:|---|
| import / 模块 | 0 | 必须 |
| 变量 / 类型 | 0 | 必须 |
| print / f-string | 0 | 必须 |
| 缩进 | 0 | 必须 |
| list / tuple / dict | 0–1 | 必须 |
| 索引 / 切片 | 1 | 必须 |
| if / for | 1 | 必须 |
| 函数 | 1 | 必须 |
| 类 / 对象 | 2 | 会用 |
| 关键字参数 | 2 | 会用 |
| with | 5 | 会用 |
| 装饰器 | 5 | 先会用 |
| 生成器 / yield | 6 | 会用 |
| 异常 | 0 | 会读 |
| `__main__` | 0 | 认识 |
| NumPy / shape | 1 | 必须 |

---

## 学习方式

1. 遇到新概念，先复制到本文件；
2. 写一个最小例子，跑通；
3. 写“在 TF 里哪里见到”；
4. 写一个自己踩过的坑；
5. 标记掌握程度，继续学 TF。

不要停下来把 Python 全学完。**遇到一个，学一个；学一个，用一个。**
