# import numpy as np
# import pandas as pd
# from datetime import datetime
# import os
#
# def save_data_to_excel(path):
#     # 创建一个大小为（7,2048）的随机数组
#     data = np.random.rand(7, 2048)
#
#     # 获取当前日期和时间
#     current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#
#     # 检查路径是否存在，不存在则创建
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     # 生成文件名，包含路径
#     filename = os.path.join(path, f"data_{current_time}.xlsx")
#
#     # 转换numpy数组为DataFrame
#     df = pd.DataFrame(data)
#
#     # 保存数据
#     df.to_excel(filename, index=False)
#     print(f"Data saved to {filename}")
#
# save_data_to_excel('./path/to/your/directory')

import numpy as np
from datetime import datetime
import os

def save_data_to_txt(path,data):
    # 创建一个大小为（7,2048）的随机数组
    #data = np.random.rand(7, 2048)

    # 获取当前日期和时间
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 检查路径是否存在，不存在则创建
    if not os.path.exists(path):
        os.makedirs(path)

    # 生成文件名，包含路径
    filename = os.path.join(path, f"data_{current_time}.txt")

    # 保存数据
    np.savetxt(filename, data)
    print(f"Data saved to {filename}")

#save_data_to_txt('./path/to/your/directory')
