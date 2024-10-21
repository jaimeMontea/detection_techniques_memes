import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from googletrans import Translator

# 设置API和语言
SRC_LANG = 'en'
TGT_LANG = 'fr'
MAX_WORKERS = 10  # 并发线程数

def translate_text(text):
    try:
        translator = Translator()
        translated_text = translator.translate(text, src=SRC_LANG, dest=TGT_LANG).text
        back_translated_text = translator.translate(translated_text, src=TGT_LANG, dest=SRC_LANG).text
        return back_translated_text  # 只返回回译后的文本
    except Exception as e:
        print(f"Error translating text: {text}\nError: {e}")
        return text

def augment_data(input_path, output_path):
    # 读取JSON文件
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    augmented_data = []

    # 创建线程池
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        # 提交翻译任务
        for item in data:
            original_text = item['text']
            futures.append(executor.submit(translate_text, original_text))
        
        # 获取翻译结果
        for future, item in zip(tqdm(futures, desc="Translating"), data):
            back_translated_text = future.result()
            augmented_item = item.copy()
            augmented_item['text'] = back_translated_text
            augmented_data.append(augmented_item)

    # 保存增强后的数据集
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(augmented_data, outfile, ensure_ascii=False, indent=4)

def main():
    input_path = 'merged_file.json'
    output_path = '3augmented_output.json'
    augment_data(input_path, output_path)

if __name__ == '__main__':
    main()

