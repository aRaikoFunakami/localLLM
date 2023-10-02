import pandas as pd



# この関数は、descriptionを処理し、新しいdescriptionを返します。
# この関数を必要に応じてカスタマイズしてください。
def process_description(description):
	# 例: descriptionを大文字に変換
	new_description = description
	return new_description

def main():
	input_file = './data/redmine_db.csv'  # 入力CSVファイル名
	output_file = 'ten_db.csv'  # 出力CSVファイル名

	# CSVファイルを読み込む
	df = pd.read_csv(input_file)

	# 最初の10行だけを処理する
	df = df.head(10)

	# 結果をCSVファイルに書き込む
	df.to_csv(output_file, index=False)

if __name__ == "__main__":
	main()
