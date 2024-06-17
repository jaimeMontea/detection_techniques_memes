import torch
from transformers import DebertaForSequenceClassification, DebertaTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import streamlit as st


from collections import OrderedDict

# Define the device where you want to place the models
device = torch.device('cpu')  # Use 'cuda' if CUDA is available

deberta_model1 = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=20)
roberta_model1 = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=20)
# Load the DeBERTa model
deberta_state_dict = torch.load("deberta_subtask1_epoch_0.pth", map_location=device)
deberta_model = deberta_model1  # Instantiate your DeBERTa model
deberta_model.load_state_dict(deberta_state_dict)

# Load the RoBERTa model
roberta_state_dict = torch.load("roberta_subtask1_epoch_0.pth", map_location=device)
roberta_model = roberta_model1  # Instantiate your RoBERTa model
roberta_model.load_state_dict(roberta_state_dict)

# Move both models to the specified device
deberta_model.to(device)
roberta_model.to(device)


# 初始化tokenizer
deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 定义模型评估函数
def evaluate_sentence(sentence):
    # 处理输入句子
    encoded_sentence_deberta = deberta_tokenizer(sentence, truncation=True, padding=True, return_tensors='pt')
    encoded_sentence_roberta = roberta_tokenizer(sentence, truncation=True, padding=True, return_tensors='pt')

    # 推断模式
    deberta_model.eval()
    roberta_model.eval()

    # 在设备上运行
    encoded_sentence_deberta = {key: val.to(device) for key, val in encoded_sentence_deberta.items()}
    encoded_sentence_roberta = {key: val.to(device) for key, val in encoded_sentence_roberta.items()}

    possible_labels = ['Appeal to authority',
 'Appeal to fear/prejudice',
 'Bandwagon',
 'Black-and-white Fallacy/Dictatorship',
 'Causal Oversimplification',
 'Doubt',
 'Exaggeration/Minimisation',
 'Flag-waving',
 'Glittering generalities (Virtue)',
 'Loaded Language',
 "Misrepresentation of Someone's Position (Straw Man)",
 'Name calling/Labeling',
 'Obfuscation, Intentional vagueness, Confusion',
 'Presenting Irrelevant Data (Red Herring)',
 'Reductio ad hitlerum',
 'Repetition',
 'Slogans',
 'Smears',
 'Thought-terminating cliché',
 'Whataboutism']
    
    # 获取预测结果
    with torch.no_grad():
        deberta_output = torch.sigmoid(deberta_model(**encoded_sentence_deberta).logits)
        roberta_output = torch.sigmoid(roberta_model(**encoded_sentence_roberta).logits)
        


    # 组合预测结果
    # combined_output = torch.max(deberta_output, roberta_output)
    combined_output = deberta_output + roberta_output

    # 返回预测结果
    threshold = 0.5  # 阈值
    labels = (combined_output > threshold).int().cpu().numpy().tolist()[0]
    output_labels = [label for label, pred in zip(possible_labels, labels) if pred == 1]
    return output_labels 

# Streamlit 应用程序
def main():
    st.title("Demo: Subtask1")

    # 添加输入框功能
    sentence = st.text_input("Please input a sentence:", "")

    # 显示预测结果
    if sentence:
        labels = evaluate_sentence(sentence)
        st.write("Predicted labels:", labels)

# 运行应用程序
if __name__ == "__main__":
    main()
