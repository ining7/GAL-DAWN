import random
import sys

# 获取命令行参数并将其转换为整数
num_of_randoms = int(sys.argv[1])

# 打开文件以进行写入操作
with open("sourceList.txt", "w") as file:
    # 生成指定数量的随机整数并将它们写入文件中
    for i in range(num_of_randoms):
        rand_int = random.randint(0, 2**31)
        file.write(str(rand_int) + "\n")

