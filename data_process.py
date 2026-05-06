import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_excel_samples(input_file, train_size=0.8, output_train='train_set.xlsx', output_test='test_set.xlsx'):
    """
    读取 Excel 文件并按比例划分训练集和测试集。

    参数:
    input_file: 输入 Excel 文件路径
    train_size: 训练集占比 (0.0 ~ 1.0)
    output_train: 训练集保存路径
    output_test: 测试集保存路径
    """
    try:
        df = pd.read_excel(input_file)
        print(f"成功读取文件：{input_file}，总样本数：{len(df)}")

        train_df, test_df = train_test_split(
            df,
            train_size=train_size,
            random_state=42,
            shuffle=True
        )

        train_df.to_excel(output_train, index=False)
        test_df.to_excel(output_test, index=False)

        print("处理完成")
        print(f"训练集已保存至：{output_train} (样本数: {len(train_df)})")
        print(f"测试集已保存至：{output_test} (样本数: {len(test_df)})")

    except Exception as e:
        print(f"发生错误: {e}")


def split_json_files(input_dir, train_size=0.8, output_train='train_set.jsonl', output_test='test_set.jsonl'):
    """
    读取文件夹中的 json 文件，并按比例划分训练集和测试集。

    参数:
    input_dir: json 文件所在文件夹
    train_size: 训练集占比 (0.0 ~ 1.0)
    output_train: 训练集 jsonl 输出路径
    output_test: 测试集 jsonl 输出路径
    """
    input_path = Path(input_dir)
    json_files = sorted(input_path.glob('*.json'))

    if not json_files:
        raise ValueError(f'未在目录中找到 json 文件: {input_dir}')

    samples = []
    for json_file in json_files:
        with json_file.open('r', encoding='utf-8') as f:
            sample = json.load(f)
            sample['sample_name'] = json_file.name
            sample['source'] = "策划报告-BadCase"
            samples.append(sample)

    train_samples, test_samples = train_test_split(
        samples,
        train_size=train_size,
        random_state=42,
        shuffle=True
    )

    with open(output_train, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(output_test, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"成功读取目录：{input_dir}，总样本数：{len(samples)}")
    print("处理完成")
    print(f"训练集已保存至：{output_train} (样本数: {len(train_samples)})")
    print(f"测试集已保存至：{output_test} (样本数: {len(test_samples)})")



def compute_accuracy(input_file):
    df = pd.read_excel(input_file)
    correct = 0
    for i in range(df.shape[0]):
        answer = df.loc[i, 'answer']
        prediction = df.loc[i, 'L2鍒嗙被缁撴灉']
        if answer == prediction:
            correct += 1
        else:
            print(i, answer, '***', prediction)

    print(correct / df.shape[0])


if __name__ == "__main__":
    # split_excel_samples(input_file='data.xlsx', train_size=0.5)
    # compute_accuracy(input_file='./shortage_analyze/data.xlsx')

    split_json_files(r'D:\Data\agent\trace\一句话BadCase', train_size=0.2, output_train='train_set.jsonl')