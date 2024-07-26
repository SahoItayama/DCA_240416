import ast
import json

filename = "parameters_txt/parameters40-200-3000a.txt"

with open(filename, "r") as file:
    # 6行目から9行目までを一つの要素として取得
    # lines = file.readlines()[5:9]
    # readだとファイル全体を文字列として取得    
    # lines = file.read()[5:9]
    lines = file.readlines()

theta_lines = lines[5:9]
c_lines = lines[9:609]
xi_lines = lines[609:]

print(len(theta_lines))

def combine_to_string(lines):
    combined_string = ''
    for i in range(len(lines)):
        combined_string += lines[i].strip() + ' '
    return combined_string

# 特定の行範囲を一つの文字列に結合（例：1行目から3行目）
# start_line = 0  # 開始行（0始まりのインデックス）
# end_line = 4
# end_line_c = 319   # 終了行（この行は含まれない）

# 指定された行範囲のデータを一つの文字列に結合

# 結合した文字列の"[", "]"を削除
combined_string = combine_to_string(theta_lines).replace('[', '').replace(']', '')
# 結合した文字列を数値のリストに変換
theta = list(map(int, combined_string.split(',')))
print(theta)
print(len(theta))

combined_string_c = combine_to_string(c_lines)
c = ast.literal_eval(combined_string_c)

combined_string_xi = combine_to_string(xi_lines)
xi = ast.literal_eval(combined_string_xi)

data = {"theta": theta, "c": c, "xi": xi}

# jsonファイルに書き込み
with open("parameters/parameters40-200-3000a.json", "w") as f:
    json.dump(data, f, indent=4)

# 指定された行範囲のデータを一つの文字列に結合
# combined_string_c = ''
# for i in range(start_line, end_line_c):
#     combined_string_c += lines_c[i].strip() + ' '
# # 結合した文字列の"[", "]"を削除
# # combined_string = combined_string.replace('[', '').replace(']', '')
# # 結合した文字列を数値のリストに変換
# # theta = list(map(int, combined_string.split(',')))
# print(combined_string_c)


# thetaの値を取得
# theta = []
# for line in lines[:4]:
#     theta.append(line.strip())
# print(theta)
    

# theta = [value.strip() for value in lines[:4]]
# print(theta)


# ファイルの各行を処理し、必要なリスト形式に変換
# data_lists = []
# for line in lines:
#     # 改行や空白を除去
#     line = line.strip()
#     if line.startswith('[') and line.endswith(']'):
#         # 文字列をリストに変換
#         list_str = line[1:-1]
#         list_values = [int(value.strip()) for value in list_str.split(',')]
#         data_lists.append(list_values)


# print(data_lists)

# print(lines)