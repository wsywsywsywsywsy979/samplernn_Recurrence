class Fibs:
    def __init__(self, n=20):
        self.a = 0
        self.b = 1
        self.n = n
    def __iter__(self): # 返回迭代器本身
        return self
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > self.n:
            raise StopIteration
        return self.a

## 调用
fibs = Fibs()
for each in fibs:
	print(each)