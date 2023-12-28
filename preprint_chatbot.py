import streamlit as st
import os, tenacity
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
from streamlit_chat import message
import boto3
import json

# Boto3 세션 및 클라이언트 설정
session = boto3.Session()
bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)

# 텍스트 임베딩 추출 함수
def get_embedding(text):
    body = json.dumps({"inputText": text})
    model_d = 'amazon.titan-embed-text-v1'
    mime_type = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=model_d, accept=mime_type, contentType=mime_type)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

# 데이터 파일 경로 설정 및 데이터 로드
folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)

if os.path.isfile(file_path):
    print(f"{file_path} 파일이 존재합니다.")
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
else:
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    data = []
    for file in txt_files:
        txt_file_path = os.path.join(folder_path, file)
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)

    df = pd.DataFrame(data, columns=['text'])
    df['embedding'] = df.apply(lambda row: get_embedding(row.text), axis=1)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

# 코사인 유사도 계산 함수
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

# 응답 후보 반환 함수
def return_answer_candidate(df, query):
    query_embedding = get_embedding(query)
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity", ascending=False).head(3)
    return top_three_doc

# 프롬프트 생성 함수
def create_prompt(df, query):
    result = return_answer_candidate(df, query)
    prompt = f"""\n\nHuman: 당신은 preprint 서비스에 대해 설명해주는 'preprint-chatbot'라는 인공지능 언어 모델입니다. 누가 당신에게 누구냐고 물으면 preprint-ai 모델이라고 답변하세요. 누가 만들었냐고 묻는다면 preprint 개발자 최은택 이라고 답변하세요. 주어진 문서와 질문을 바탕으로 질문에 대한 답변을 간결하고 평문으로 요약해주세요.
    문서는 다음과 같습니다.
            doc 1 :""" + str(result.iloc[0]['text']) + """
            doc 2 :""" + str(result.iloc[1]['text']) + """
            doc 3 :""" + str(result.iloc[2]['text']) + """
    자기 소개 할 때는 문서에 대한 요약을 작성하지마세요. 질문과 상관없는 답은 해서는 안 됩니다.
    주어진 문서로부터 질문에 대한 답변을 평문으로 작성해주세요. 절대로 문서 3개를 언급하지마세요. 질문에 대한 답변만 작성하세요.
    
    질문: """ + str(query) + """\n\nAssistant:"""
    return prompt

# 응답 생성 함수
def generate_response(prompt):
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 300,
        "temperature": 0.1,
        "top_p": 0.9,
    })
    modelId = 'anthropic.claude-v2:1'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    return response_body.get('completion')

# Streamlit UI 구성
# st.image('images/ask_me_chatbot_logo.png')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('Preprint에 대해 물어보세요!', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    prompt = create_prompt(df, user_input)
    chatbot_response = generate_response(prompt)
    st.session_state['past'].append(user_input)
    st.session_state["generated"].append(chatbot_response)

if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
