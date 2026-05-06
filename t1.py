import pandas as pd
from sklearn.model_selection import train_test_split


def split_excel_samples(input_file, train_size=0.8, output_train='train_set.xlsx', output_test='test_set.xlsx'):
    """
    读取 Excel 文件并按比例划分训练集和测试集

    参数:
    input_file: 输入的 Excel 文件路径
    train_size: 训练集占比 (0.0 ~ 1.0)
    output_train: 训练集保存路径
    output_test: 测试集保存路径
    """
    try:
        # 1. 读取 Excel 文件
        # pandas 默认将第一行识别为列名（header=0）
        df = pd.read_excel(input_file)
        print(f"成功读取文件：{input_file}，总样本数：{len(df)}")

        # 2. 划分数据集
        # random_state 保证每次运行划分的结果一致
        # shuffle=True 会在划分前打乱样本顺序
        train_df, test_df = train_test_split(
            df,
            train_size=train_size,
            random_state=42,
            shuffle=True
        )

        # 3. 分别保存为 Excel
        # index=False 表示不保存行索引
        train_df.to_excel(output_train, index=False)
        test_df.to_excel(output_test, index=False)

        print(f"处理完成！")
        print(f"训练集已保存至：{output_train} (样本数: {len(train_df)})")
        print(f"测试集已保存至：{output_test} (样本数: {len(test_df)})")

    except Exception as e:
        print(f"发生错误: {e}")


def compute_accuracy(input_file):
    df = pd.read_excel(input_file)
    correct = 0
    for i in range(df.shape[0]):
        answer = df.loc[i, 'answer']
        prediction = df.loc[i, 'L2分类结果']
        if answer == prediction:
            correct += 1
        else:
            print(i, answer, '***', prediction)

    print(correct/df.shape[0])



if __name__ == "__main__":
    # split_excel_samples(input_file='data.xlsx', train_size=0.5)
    compute_accuracy(input_file='./shortage_analyze/data_归因分析结果.xlsx')